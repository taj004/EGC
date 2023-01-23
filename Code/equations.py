import numpy as np
import porepy as pp
import scipy.sparse as sps


def rho(p, temp):
    """Constitutive law between density and pressure"""
    # reference density
    rho_f = 1000

    # Pressure related variable
    c = 1.0e-9  # compresibility [1/Pa]
    p_ref = 1000  # reference pressure [Pa]

    # Temperatrue related parameters
    beta = 4e-4  # Fluid thermal expansion [1/K]
    temp_ref = 573.15  # [K]
    
    # Fluid density (linearised for stability purposes)
    density = rho_f * (1 + c * (p - p_ref) - beta * (temp - temp_ref))

    return density


def to_vec(gb, param_kw, param, size_type="cells", to_ad=False):
    """Make a vector of a parameter param, from a param_kw"""

    if size_type == "cells":
        vec = np.zeros(gb.num_cells())
    elif size_type == "faces":
        vec = np.zeros(gb.num_faces())
    else:
        raise ValueError("Unknown size type")
    # end if-else

    val = 0
    for g, d in gb:

        if size_type == "cells":
            num_size = g.num_cells
        else:
            num_size = g.num_faces
        # end if-else

        # Extract from the dictionary
        x = d[pp.PARAMETERS][param_kw][param]

        if isinstance(x, np.ndarray) is False:
            x *= np.ones(num_size)
        # end if

        # Indices
        inds = slice(val, val + num_size)
        # Store values and update for the next grid
        vec[inds] = x.copy()
        val += num_size
    # end g,d-loop

    if to_ad:
        return pp.ad.Array(vec)
    else:
        return vec


#%% Assemble the non-linear equations


def gather(gb, equation_manager, iterate=False):

    """
    Collect and discretize equations on a PorePy grid bucket

    Parameters:
    ----------
        gb: A PorePy grid bucket
        equation_manager: A PorePy equation manager that keeps the
                      information about the equations
        iterate: bool, whether the equations are updated in the Newton iterations (True),
                   or formed at the start of a time step (False).

    """

    # Keywords:
    mass_kw = "mass"
    flow_kw = "flow"
    transport_kw = "transport"
    chemistry_kw = "chemistry"

    # Primary variables:
    pressure = "pressure"
    tot_var = "T"  # T: The total concentration of the primary components
    log_var = (
        "log_X"  # log_c: Logarithm of C, the aqueous part of the total concentration
    )
    minerals = "minerals"
    tracer = "passive_tracer"
    temperature = "temperature"

    mortar_pressure = "mortar_pressure"
    mortar_tot_var = "mortar_transport"
    mortar_tracer = "mortar_tracer"
    mortar_temperature_convection = "mortar_temperature_convection"
    mortar_temperature_conduction = "mortar_temperature_conduction"

    # Loop over the gb to get some information
    # (more appropriate approaches exist)
    for g, d in gb:

        # Get data that are necessary for later use
        if g.dim == gb.dim_max():
            data_transport = d[pp.PARAMETERS][transport_kw]
            data_chemistry = d[pp.PARAMETERS][chemistry_kw]
            data_prev_time = d[pp.PARAMETERS]["previous_time_step"]
            data_prev_newton = d[pp.PARAMETERS]["previous_newton_iteration"]
            data_grid = d[pp.PARAMETERS]["grid_params"]
        # end_if
    # end g,d-loop

    # The list of grids and edges
    grid_list = data_grid["grid_list"]
    edge_list = data_grid["edge_list"]

    # Ad representations of the primary variables.
    p = equation_manager.merge_variables([(g, pressure) for g in grid_list])

    T = equation_manager.merge_variables([(g, tot_var) for g in grid_list])

    log_X = equation_manager.merge_variables([(g, log_var) for g in grid_list])

    precipitate = equation_manager.merge_variables([(g, minerals) for g in grid_list])

    passive_tracer = equation_manager.merge_variables([(g, tracer) for g in grid_list])

    temp = equation_manager.merge_variables([(g, temperature) for g in grid_list])

    # The fluxes over the interfaces
    if len(edge_list) > 0:
        # Flow
        v = equation_manager.merge_variables([e, mortar_pressure] for e in edge_list)

        # Transport
        eta = equation_manager.merge_variables([e, mortar_tot_var] for e in edge_list)

        # Passive tracer
        eta_tracer = equation_manager.merge_variables(
            [e, mortar_tracer] for e in edge_list
        )

        # Temperature
        w = equation_manager.merge_variables(
            [e, mortar_temperature_convection] for e in edge_list
        )
        q = equation_manager.merge_variables(
            [e, mortar_temperature_conduction] for e in edge_list
        )
    # end if

    #%% Define equilibrium equations for Ad

    # Cell-wise chemical values
    cell_S = data_chemistry["cell_stoic_coeff_S"]
    cell_E = data_chemistry["cell_stoic_coeff_E"]
    cell_equil_comp = data_chemistry["cell_equilibrium_constants_comp"]
    cell_equil_prec = data_chemistry["cell_equilibrium_constants_prec"]

    def equilibrium_all_cells(total, log_primary, precipitated_species):
        """
        Residual form of the equilibrium problem,
        posed for all cells together.
        """

        # The secondary species
        secondary_C = cell_equil_comp * pp.ad.exp(cell_S * log_primary)
        eq = (
            pp.ad.exp(log_primary)
            + cell_S.transpose() * secondary_C
            + cell_E.transpose() * precipitated_species
            - total
        )
        return eq

    # Wrap the equilibrium residual into an Ad function.
    equil_ad = pp.ad.Function(equilibrium_all_cells, "equil")

    # Finally, make it into an Expression which can be evaluated.
    equilibrium_eq = equil_ad(T, log_X, precipitate)
    equilibrium_eq.set_name("equilibrium")

    #%% The flow equation

    # The divergence
    div = data_grid["divergence_single"]

    # A projection between subdomains and mortar grids
    if len(edge_list) > 0:
        mortar_projection = data_grid["mortar_projection_single"]
    # end if

    # # Time step
    dt = data_transport["time_step"]

    # BC for flow
    bound_flux = pp.ad.BoundaryCondition(flow_kw, grid_list)
    # The boundary condition for the temperature
    bc_temp = pp.ad.BoundaryCondition(keyword=temperature, grids=grid_list)

    if iterate:  # Newton iterations
        mass_density = pp.ad.MassMatrixAd(keyword=flow_kw, grids=grid_list)

        mass_density_prev = data_prev_time["mass_density_prev"]  # previous time step
        p_prev = data_prev_time["p_prev"]  # previous time step
        temp_prev = data_prev_time["temp_prev"]
    else:
        # The pressure at the previous time step
        p_prev = p.previous_timestep()
        temp_prev = temp.previous_timestep()

        # Acculmuation for the pressure equation
        mass_density = pp.ad.MassMatrixAd(keyword=flow_kw, grids=grid_list)
        mass_density_prev = pp.ad.MassMatrixAd(keyword=flow_kw, grids=grid_list)

        # Store for the Newton calculations
        data_prev_time["p_prev"] = p_prev
        data_prev_time["mass_density_prev"] = mass_density_prev
    # end if-else

    # Wrap the constitutive law into ad, and construct the ad-wrapper of the PDE
    rho_ad = pp.ad.Function(rho, "density")
    rho_ad_main = rho_ad(p, temp_prev)

    # Build the flux term.
    
    # Scale the fluxes by density.
    # The density is a cell-centerd variable,
    # so we need to average it onto the faces
    # NOTE: the values in the darcy flux for this variable have to be +1
    # to ensure that the scaling becomes correct
    upwind_weight = pp.ad.UpwindAd(keyword=flow_kw, grids=grid_list)

    # The boundary conditions
    rho_ad_bc = rho_ad(bound_flux, bc_temp)

    # If want to run the script, but on a different grid 
    # and with choice of boundary conditions,
    # you might need to change the minus to a plus
    rho_on_face = (
        upwind_weight.upwind * rho_ad_main
        - upwind_weight.bound_transport_dir * rho_ad_bc
        - upwind_weight.bound_transport_neu * rho_ad_bc
    )

    # Ad wrapper of Mpfa discretization
    # mpfa = pp.ad.TpfaAd(flow_kw, grids=grid_list)
    mpfa = pp.ad.MpfaAd(flow_kw, grids=grid_list)

    # NOTE: Numerous reasons can cause a "need csr matrix" error-message.
    # If you get one, some debugging ideas:
    #   - test on a unfractured grid    
    #   - ensure all mpfa-related discretization matrices have a csr-format
  
    # If non of these work, either use tpfa 
    # (this increases simulation run time and 
    # will only cause minor (decimal) changes in the results) 
    # or contact me

    # The interior and boundary fluxes
    interior_flux = mpfa.flux * p
    boundary_flux = mpfa.bound_flux * bound_flux

    # The subdomain Darcy flux
    full_flux = interior_flux + boundary_flux

    # Multiply the flux and the density
    full_density_flux = rho_on_face * full_flux

    # Add the fluxes from the interface to higher dimension,
    # the source accounting for fluxes to the lower dimansions
    # and multiply by the divergence
    if len(edge_list) > 0:

        # Include flux from the interface
        full_flux += mpfa.bound_flux * mortar_projection.mortar_to_primary_int * v

        # Tools to include the density in the source term
        upwind_coupling_weight = pp.ad.UpwindCouplingAd(
            keyword=flow_kw, edges=edge_list
        )
        trace = data_grid["trace_single"]

        up_weight_primary = upwind_coupling_weight.upwind_primary
        up_weight_secondary = upwind_coupling_weight.upwind_secondary

        # Map the density in the higher- and lower-dimensions
        # onto the interfaces
        high_to_low = (
            up_weight_primary
            * mortar_projection.primary_to_mortar_avg
            * trace.trace
            * rho_ad_main
        )

        low_to_high = (
            up_weight_secondary
            * mortar_projection.secondary_to_mortar_avg
            * rho_ad_main
        )

        # The density at the interface
        mortar_density = (high_to_low + low_to_high) * v

        # Include the mortar density in the flux towards \Omega_h
        full_density_flux += (
            mpfa.bound_flux * mortar_projection.mortar_to_primary_int * mortar_density
        )

        # The source term for the flux to \Omega_l
        sources_from_mortar = mortar_projection.mortar_to_secondary_int * mortar_density

        conservation = div * full_density_flux - sources_from_mortar

    else:
        conservation = div * full_density_flux
    # end if

    # Construct the conservation equation
    density_wrap = (
        mass_density.mass * rho_ad_main
        - mass_density_prev.mass * rho_ad(p_prev, temp_prev)
    ) / dt + conservation
    if len(edge_list) > 0:

        pressure_trace_from_high = mortar_projection.primary_to_mortar_avg * (
            mpfa.bound_pressure_cell * p
            + mpfa.bound_pressure_face
            * (mortar_projection.mortar_to_primary_int * v + bound_flux)
        )

        pressure_from_low = mortar_projection.secondary_to_mortar_avg * p

        robin = pp.ad.RobinCouplingAd(flow_kw, edge_list)

        # The interface flux, lambda
        interface_flux = (
            robin.mortar_discr * (pressure_trace_from_high - pressure_from_low) + v
        )
    # end if

    # Make equations, feed them to the AD-machinery and discretize
    if len(edge_list) > 0:
        # Conservation equation
        density_wrap.discretize(gb)
        density_wrap.set_name("fluid_conservation")
        # Flux over the interface
        interface_flux.discretize(gb)
        interface_flux.set_name("interface_flux")
    else:
        density_wrap.discretize(gb)
        density_wrap.set_name("fluid_conservation")
    # end if-else

    # Store the flux
    data_prev_newton["AD_full_flux"] = full_flux

    if len(edge_list) > 0:
        data_prev_newton.update({"AD_edge_flux": v})
    # end if

    #%% The Passive tracer

    # It is similar to the solute transport equations, which is studied
    # in detail below.
    # The reason I placed the construction of the tracer
    # above the solute transport equation is to avoid having to "redefine"
    # the div, mortar_projection, etc operators

    # Upwind discretization for advection.
    upwind_tracer = pp.ad.UpwindAd(keyword=tracer, grids=grid_list)
    if len(edge_list) > 0:
        upwind_tracer_coupling = pp.ad.UpwindCouplingAd(keyword=tracer, edges=edge_list)
    # end if

    # Ad wrapper of boundary conditions
    bc_tracer = pp.ad.BoundaryCondition(keyword=tracer, grids=grid_list)
    # Mass matrix for accumulation
    mass_tracer = pp.ad.MassMatrixAd(tracer, grid_list)

    if iterate:  # Newton-iteration
        mass_tracer_prev = data_prev_time["mass_tracer_prev"]
        tracer_prev = data_prev_time["tracer_prev"]
    else:  # We are interested in "constructing" the equations

        mass_tracer_prev = pp.ad.MassMatrixAd(tracer, grid_list)
        tracer_prev = passive_tracer.previous_timestep()

        # Store for Newton iterations
        data_prev_time["tracer_prev"] = tracer_prev
        data_prev_time["mass_tracer_prev"] = mass_tracer_prev
    # end if-else

    tracer_adv = (
        (upwind_tracer.upwind * passive_tracer) * full_flux
        - upwind_tracer.bound_transport_dir * full_flux * bc_tracer
        - upwind_tracer.bound_transport_neu * bc_tracer
    )

    tracer_wrapper = (
        (mass_tracer.mass * passive_tracer - mass_tracer_prev.mass * tracer_prev) / dt
        # Advection
        + div * tracer_adv
    )

    # Add the projections
    if len(edge_list) > 0:
        tracer_wrapper -= div * (
            upwind_tracer.bound_transport_neu
            * mortar_projection.mortar_to_primary_int
            * eta_tracer
        )

        tracer_wrapper -= mortar_projection.mortar_to_secondary_int * eta_tracer
    # end if

    # Tracer over the interface
    if len(edge_list) > 0:

        # Some tools we need
        upwind_tracer_coupling_primary = upwind_tracer_coupling.upwind_primary
        upwind_tracer_coupling_secondary = upwind_tracer_coupling.upwind_secondary

        # First project the concentration from high to low
        trace_of_tracer = trace.trace * passive_tracer

        high_to_low_tracer = (
            upwind_tracer_coupling_primary
            * mortar_projection.primary_to_mortar_avg
            * trace_of_tracer
        )

        # Next project concentration from lower onto higher dimension
        low_to_high_tracer = (
            upwind_tracer_coupling_secondary
            * mortar_projection.secondary_to_mortar_avg
            * passive_tracer
        )

        # Finally we have the transport over the interface equation
        tracer_over_interface_wrapper = (
            upwind_tracer_coupling.mortar_discr * eta_tracer
            - (high_to_low_tracer + low_to_high_tracer) * v
        )
    # end if

    if len(edge_list) > 0:
        tracer_wrapper.discretize(gb)
        tracer_wrapper.set_name("passive_tracer")
        tracer_over_interface_wrapper.discretize(gb)
        tracer_over_interface_wrapper.set_name("tracer_interface_flux")
    else:
        tracer_wrapper.discretize(gb)
        tracer_wrapper.set_name("passive_tracer")
    # end if-else

    #%% The temperatue equtaion.

    # Upwind discretization for the convection
    upwind_temp = pp.ad.UpwindAd(keyword=temperature, grids=grid_list)
    heat_capacity = pp.ad.MassMatrixAd(keyword=temperature, grids=grid_list)

    if iterate:
        heat_capacity_prev = data_prev_time["heat_capacity_prev"]
        temp_prev = data_prev_time["temp_prev"]
    else:
        heat_capacity_prev = pp.ad.MassMatrixAd(keyword=temperature, grids=grid_list)
        # Store for later use
        data_prev_time["temp_prev"] = temp_prev
        data_prev_time["heat_capacity_prev"] = heat_capacity_prev
    # end if-else

    # Temperature "scaled" by density and specific heat capacites
    specific_heat_cells = to_vec(
        gb, param_kw=temperature, param="specific_heat_fluid", to_ad=True
    )
    specific_heat_faces = to_vec(
        gb,
        param_kw=temperature,
        param="specific_heat_fluid",
        size_type="faces",
        to_ad=True,
    )

    temp_density = (
        temp
        * specific_heat_cells
        * rho_ad(p.previous_iteration(), temp.previous_iteration())
    )
    bound_temp = bc_temp * rho_ad_bc * specific_heat_faces

    # Convection term
    convection = (
        full_flux * (upwind_temp.upwind * temp_density)
        - upwind_temp.bound_transport_dir * full_flux * bound_temp
        - upwind_temp.bound_transport_neu * bound_temp
    )

    # Conduction
    conduction_mpfa = pp.ad.MpfaAd(temperature, grid_list) # pp.ad.TpfaAd(temperature, grids=grid_list)
    conduction = conduction_mpfa.flux * temp + conduction_mpfa.bound_flux * bc_temp

    # Cell-wise temperature equation
    temp_wrap = (
        heat_capacity.mass * temp - heat_capacity_prev.mass * temp_prev
    ) / dt + div * (convection + conduction)

    # Projections
    if len(edge_list) > 0:

        # Edge conduction
        temp_wrap += (
            div
            * conduction_mpfa.bound_flux
            * mortar_projection.mortar_to_primary_int
            * q
        )
        temp_wrap -= mortar_projection.mortar_to_secondary_int * q

        # Edge convection
        temp_wrap -= (
            div
            * upwind_temp.bound_transport_neu
            * mortar_projection.mortar_to_primary_int
            * w
        )
        temp_wrap -= mortar_projection.mortar_to_secondary_int * w
    # end if

    # convection over interface
    if len(edge_list) > 0:
        # Upstream convection
        convection_coupling = pp.ad.UpwindCouplingAd(
            keyword=temperature, edges=edge_list
        )

        high_to_low_convection = (
            convection_coupling.upwind_primary
            * mortar_projection.primary_to_mortar_avg
            * trace.trace
            * temp_density
        )

        low_to_high_convection = (
            convection_coupling.upwind_secondary
            * mortar_projection.secondary_to_mortar_avg
            * temp_density
        )

        interface_convection = (
            convection_coupling.mortar_discr * w
            - (high_to_low_convection + low_to_high_convection) * v
        )

        # Conductive flux
        temp_from_high = mortar_projection.primary_to_mortar_avg * (
            conduction_mpfa.bound_pressure_cell * temp
            + conduction_mpfa.bound_pressure_face
            * (mortar_projection.mortar_to_primary_int * q + bc_temp)
        )
        temp_from_low = mortar_projection.secondary_to_mortar_avg * temp

        robin_temp = pp.ad.RobinCouplingAd(keyword=temperature, edges=edge_list)

        conduction_over_interface = q + robin_temp.mortar_discr * (
            temp_from_high - temp_from_low
        )
    # end if

    # Discretize the equations
    if len(edge_list) > 0:
        temp_wrap.discretize(gb)
        interface_convection.discretize(gb)
        conduction_over_interface.discretize(gb)
    else:
        temp_wrap.discretize(gb)
    # end if-else
    #%% Next, the (solute) transport equations.

    # Upwind discretization for advection.
    upwind = pp.ad.UpwindAd(keyword=transport_kw, grids=grid_list)

    # Ad wrapper of boundary conditions
    bc_c = pp.ad.BoundaryCondition(keyword=transport_kw, grids=grid_list)

    # Divergence operator. Acts on fluxes of aquatic components only.
    div = data_grid["divergence_several"]

    # Mortar projection between subdomains and mortar grids
    if len(edge_list) > 0:
        mortar_projection = data_grid["mortar_projection_several"]
    # end if

    # Conservation applies to the linear form of the total consentrations.
    # Make a function which carries out both conversion and summation over primary
    # and secondary species.
    def log_to_linear(c: pp.ad.Ad_array) -> pp.ad.Ad_array:
        return pp.ad.exp(c) + cell_S.T * cell_equil_comp * pp.ad.exp(cell_S * c)

    # Wrap function in an Ad function, ready to be parsed
    log2lin = pp.ad.Function(log_to_linear, "")
    log_c = log2lin(log_X)

    # For the accumulation term, we need T and the porosity at
    # the previous time step. These should be fixed in the Newton-iterations
    mass = pp.ad.MassMatrixAd(mass_kw, grid_list)
    if iterate:  # Newton-iteration
        mass_prev = data_prev_time["mass_prev"]
        T_prev = data_prev_time["T_prev"]
    else:  # We are interested in "constructing" the equations

        # Fixed mass matrix
        mass_prev = pp.ad.MassMatrixAd(mass_kw, grid_list)

        # The solute solution at time step n,
        # ie. the one we need to use to find the solution at time step n+1
        T_prev = T.previous_timestep()

        # Store for Newton iterations
        data_prev_time["T_prev"] = T_prev
        data_prev_time["mass_prev"] = mass_prev
    # end if-else

    # We need four terms for the solute transport equation:
    # 1) Accumulation
    # 2) Advection
    # 3) Boundary condition for inlet
    # 4) boundary condition for outlet.

    # The upwind discretization matrices are calculated so that
    # they are 1 at the flux subdomain indices, and not scaled by the flux at all.
    # So it is not in complete accordance with the description of the paper.
    # If you want to a complaint about this, contact Eirik Keilegavlen.

    # The fluxes computed above, in the flow part as
    # full_flux = mpfa * p + bound mpfa * p_bound + ...
    # Expand the flux vector to calculate the avective flux
    expanded_flux = pp.ad.Matrix(data_grid["extend_flux"]) * full_flux

    all_2_aq = data_grid["all_2_aquatic"]
    advection = (
        expanded_flux * (upwind.upwind * all_2_aq * log_c)
        - upwind.bound_transport_dir * expanded_flux * bc_c
        - upwind.bound_transport_neu * bc_c
    )

    transport = (
        mass.mass * T - mass_prev.mass * T_prev
    ) / dt + all_2_aq.transpose() * div * advection

    # Add the projections
    if len(edge_list) > 0:
        all_2_aquatic_at_interface = data_grid["all_2_aquatic_at_interface"]
        transport -= div * (
            upwind.bound_transport_neu
            * mortar_projection.mortar_to_primary_int
            * all_2_aquatic_at_interface.transpose()
            * eta
        )
        # The trace operator
        trace = data_grid["trace_several"]

        transport -= (
            mortar_projection.mortar_to_secondary_int
            * all_2_aquatic_at_interface.transpose()
            * eta
        )
    # end if

    #%% Transport over the interface
    if len(edge_list) > 0:
        upwind_coupling = pp.ad.UpwindCouplingAd(keyword=transport_kw, edges=edge_list)
        # Some tools we need
        upwind_coupling_primary = upwind_coupling.upwind_primary
        upwind_coupling_secondary = upwind_coupling.upwind_secondary

        expanded_v = pp.ad.Matrix(data_grid["extend_edge_flux"]) * v
        # First project the concentration from high to low
        # At the higher-dimensions, we have both fixed
        # and aqueous concentration. Only the aqueous
        # concentrations are transported
        high_to_low_trans = (
            upwind_coupling_primary
            * mortar_projection.primary_to_mortar_avg
            * trace.trace
            * log_c
        )

        # Next project concentration from lower onto higher dimension
        # At the lower-dimension, we also have both
        # fixed and aqueous concentration
        low_to_high_trans = (
            upwind_coupling_secondary
            * mortar_projection.secondary_to_mortar_avg
            * log_c
        )

        # Finally we have the transport over the interface equation
        transport_over_interface = (
            upwind_coupling.mortar_discr * eta
            - (high_to_low_trans + low_to_high_trans) * expanded_v
        )
    # end if

    if len(edge_list) > 0:
        transport.discretize(gb)
        transport.set_name("solute_transport")
        transport_over_interface.discretize(gb)
        transport_over_interface.set_name("interface_transport")
    else:
        transport.discretize(gb)
        transport.set_name("solute_transport")
    # end if-else

    #%% The last equation is mineral precipitation-dissolution

    def F_matrix(x1, x2):
        """
        Construct diagonal matrices, representing active and inactive sets
        """
        f = x1 - x2 > 0
        f = f.astype(float)
        Fa = sps.diags(f)  # Active
        Fi = sps.diags(1 - f)  # Inactive
        return Fa, Fi

    def phi_min(mineral_conc, log_primary):
        """
        Evaluation of the min function
        """
        sec_conc = 1 - cell_equil_prec * pp.ad.exp(cell_E * log_primary)
        Fa, Fi = F_matrix(mineral_conc.val, sec_conc.val)
        eq = Fa * sec_conc + Fi * mineral_conc
        return eq

    ad_min_1 = pp.ad.Function(phi_min, "")
    mineral_eq = ad_min_1(precipitate, log_X)
    mineral_eq.set_name("dissolution_and_precipitation")

    #%% The final step is to feed the equations to the equation manager
    #   and return the non-linear equations
    equation_manager.equations.clear()
    # breakpoint()
    if len(edge_list) > 0:
        equation_manager.equations = {
            "fluid_conservation": density_wrap,
            "interface_flux": interface_flux,
            "solute_transport": transport,
            "interface_transport": transport_over_interface,
            "equilibrium": equilibrium_eq,
            "dissolution_and_precipitation": mineral_eq,
            "passive_tracer": tracer_wrapper,
            "tracer_interface_flux": tracer_over_interface_wrapper,
            "temperature": temp_wrap,
            "convection_interface": interface_convection,
            "conduction_interface": conduction_over_interface,
        }
    else:
        equation_manager.equations = {
            "fluid_conservation": density_wrap,
            "solute_transport": transport,
            "equilibrium": equilibrium_eq,
            "dissolution_and_precipitation": mineral_eq,
            "passive_tracer": tracer_wrapper,
            "temperature": temp_wrap,
        }
    # end if-else

    return equation_manager
