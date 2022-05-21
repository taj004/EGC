"""
@author: uw
"""

import numpy as np
import porepy as pp
import scipy.sparse as sps


def repeat(v, reps, gb, dof_manager):
    """
    Repeat a vector v, reps times
    Main target is the flux vectors, in the transport processes  
    """
    # Currently only for ad_arrays
    if isinstance(v, np.ndarray):
        raise ValueError("Can only repeat Ad-arrays")
    # end if    
    
    # Get expression for evaluation and check attributes 
    expression_v = v.evaluate(dof_manager)
    if isinstance(expression_v, np.ndarray):
        num_v = expression_v
    elif hasattr(expression_v, "val"):
        num_v = expression_v.val
    else :
        raise ValueError("Cannot repeat this vector")
    # end if-else
    
    num_v_reps = np.repeat(num_v, reps)
    
    # Wrap the array in Ad.
    # Note that we return absolute value of the expanded vectors
    # The reason is these vectors are ment to use as scale values 
    # in the solute transport equation. 
    # The signs are handled in the darcy_flux  
    ad_reps = pp.ad.Array(np.abs(num_v_reps))

    return  ad_reps

def remove_frac_face_flux(full_flux, gb, dof_manager):
    """Put the fluxes at the fracture faces to zero"""
    
    num_flux = full_flux.evaluate( dof_manager)
    if hasattr(num_flux, "val"):
        num_flux = num_flux.val
    # end if
    val = 0
    for g, d in gb:
        
        # Get the fracture faces 
        fracture_faces = g.tags["fracture_faces"]
        
        # Remove the contribution on the faces directly
        is_fracture_faces = np.where(fracture_faces == True)[0]
        
        # To remove the flux at fracture fraces in the lower dimensions,
        # we need to correct the fracture face indexing
     
        correct_fracture_faces = is_fracture_faces + val
        val += g.num_faces
        if len(is_fracture_faces) > 0:    
            num_flux[correct_fracture_faces] = 0
        # end if

    # end g,d-loop
    
    return pp.ad.Array(num_flux)

def rho(p):
    """
    Constitutive law between density and pressure 
    """
    # reference density 
    rho_f = 1.0e3 
    
    # Pressure related variable
    c = 1.0e-9 # compresibility
    p_ref = 1e3 # reference pressure [Pa]
    
    if isinstance(p, np.ndarray) or isinstance(p, int): # The input variables is a np.array
        density = rho_f * np.exp(
            c * (p - p_ref) 
            )
    else: # For the mass conservation equation for the fluid
        density = rho_f * pp.ad.exp(
            c * (p - p_ref) 
            ) 
    # end if-else
    return density

#%% Assemble the non-linear equations
    
def gather(gb, 
            dof_manager,
            equation_manager,
            iterate = False
            ):
    
    """
    Collect and discretize equations on a GB 
    
    Parameters
    ----------
    gb : grid bucket
    dof_manager : dof_manager for ...
    equation_manager: an equation manager that keeps the 
                      information about the equations
    iterate : bool, whether the equations are updated in the Newton iterations (True),
                    or formed at the start of a time step (False).  
    """    
    
    # Keywords:
    mass_kw = "mass"
    flow_kw = "flow"
    transport_kw = "transport"
    chemistry_kw = "chemistry"
    
    # Primary variables:
    pressure = "pressure"
    tot_var = "T" # T: The total concentration of the primary components
    log_var = "log_X" # log_c: Logarithm of C, the aqueous part of the total concentration
    minerals = "minerals"
    tracer = "passive_tracer"
    
    mortar_pressure = "mortar_pressure"
    mortar_tot_var = "mortar_transport"   
    mortar_tracer = "mortar_tracer"
    
    # Loop over the gb
    for g,d in gb:
        
        # Get data that are necessary for later use
        if g.dim == gb.dim_max():
            data_transport = d[pp.PARAMETERS][transport_kw]
            data_chemistry = d[pp.PARAMETERS][chemistry_kw]
            data_prev_time = d[pp.PARAMETERS]["previous_time_step"]
            data_prev_newton = d[pp.PARAMETERS]["previous_newton_iteration"]      
            data_grid = d[pp.PARAMETERS]["grid_params"]
            
            # The number of components         
            num_aq_components = d[pp.PARAMETERS][transport_kw]["num_components"]
            num_components = d[pp.PARAMETERS][mass_kw]["num_components"]
        # end_if
    # end g,d-loop

    # The list of grids and edges
    grid_list = data_grid["grid_list"]
    edge_list = data_grid["edge_list"]
     
    # Ad representations of the primary variables.
    p = equation_manager.merge_variables(
        [(g, pressure) for g in grid_list]
        )
    
    T = equation_manager.merge_variables(
        [(g, tot_var) for g in grid_list]
        )
    
    log_X = equation_manager.merge_variables(
        [(g, log_var) for g in grid_list]
        )
    
    precipitate = equation_manager.merge_variables(
        [(g, minerals) for g in grid_list]
        )
    
    passive_tracer = equation_manager.merge_variables(
        [(g, tracer) for g in grid_list]
        )
    
      # The fluxes over the interfaces
    if len(edge_list) > 0:
        # Flow
        lam = equation_manager.merge_variables([e, mortar_pressure] for e in edge_list)
        
        # Transport
        eta = equation_manager.merge_variables([e, mortar_tot_var] for e in edge_list)
        
        # Passive tracer
        eta_tracer = equation_manager.merge_variables([e, mortar_tracer] for e in edge_list)
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
            pp.ad.exp(log_primary) + 
            cell_S.transpose() * secondary_C + 
            cell_E.transpose() * precipitated_species - 
            total
              ) 
        return eq
    
    # Wrap the equilibrium residual into an Ad function.
    equil_ad = pp.ad.Function(equilibrium_all_cells, "equil")
    
    # Finally, make it into an Expression which can be evaluated.
    equilibrium_eq = equil_ad(T, log_X, precipitate)
    equilibrium_eq.set_name("equilibrium")
    #%% The flow equation 
   
    # Compute the second order tensor
    for g,d in gb:
        K = d[pp.PARAMETERS][flow_kw]["permeability"] # The permeability
        d[pp.PARAMETERS][flow_kw].update({"second_order_tensor": pp.SecondOrderTensor(K)})
    # end g,d-loop
    
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
           
    if iterate : # Newton iterations
        mass_density = pp.ad.MassMatrixAd(keyword=flow_kw, grids=grid_list)
    
        mass_density_prev = data_prev_time["mass_density_prev"] # previous time step
        p_prev = data_prev_time["p_prev"] # previous time step
    else:
        # The pressure at the previous time step
        p_prev = p.previous_timestep()

        # Acculmuation for the pressure equation
        mass_density = pp.ad.MassMatrixAd(keyword=flow_kw, grids=grid_list)
        mass_density_prev = pp.ad.MassMatrixAd(keyword=flow_kw, grids=grid_list)
        
        # Store for the Newton calculations
        data_prev_time["p_prev"] = p_prev
        data_prev_time["mass_density_prev"] = mass_density_prev
    # end if-else

    # Wrap the constitutive law into ad, and construct the ad-wrapper of the PDE 
    rho_ad = pp.ad.Function(rho, "density")
    rho_ad_main = rho_ad(p) 
    
    # Build the flux term. 
    # Remember that we need to multiply the fluxes by density. 
    # The density is a cell-centerd variable,so we need to average it onto the faces 
    # NOTE: the values in the darcy flux for this variable have to be +1
    # to ensure that the scaling becomes correct
    upwind_weight = pp.ad.UpwindAd(keyword=flow_kw, grids=grid_list)  
     
    # The boundary conditions 
    rho_ad_bc = rho_ad(bound_flux)
    
    rho_on_face = (
        upwind_weight.upwind * rho_ad_main
        + upwind_weight.bound_transport_dir * rho_ad_bc
        + upwind_weight.bound_transport_neu * rho_ad_bc
        ) 

    # Ad wrapper of Mpfa discretization
    mpfa = pp.ad.TpfaAd(flow_kw, grids=grid_list) 
    
    # The interior and boundary fluxes
    interior_flux = mpfa.flux * p 
    boundary_flux = mpfa.bound_flux * bound_flux
    
    # The Darcy flux (in the higher dimension)
    full_flux = interior_flux + boundary_flux 
    
    # Multiply the flux and the density
    full_density_flux = rho_on_face * full_flux
    
    # Add the fluxes from the interface to higher dimension,
    # the source accounting for fluxes to the lower dimansions
    # and multiply by the divergence
    if len(edge_list) > 0:
        
        # Include flux from the interface
        full_flux += mpfa.bound_flux * mortar_projection.mortar_to_primary_int * lam
      
        # Tools to include the density in the source term 
        upwind_coupling_weight = pp.ad.UpwindCouplingAd(keyword=flow_kw, edges=edge_list)
        trace = data_grid["trace_single"]
              
        #up_weight_flux = upwind_coupling_weight.flux
        up_weight_primary = upwind_coupling_weight.upwind_primary
        up_weight_secondary = upwind_coupling_weight.upwind_secondary
        
        # Map the density in the higher- and lower-dimensions 
        # onto the interfaces
        high_to_low = ( 
            up_weight_primary * 
            mortar_projection.primary_to_mortar_int * 
            trace.trace * rho_ad_main
            ) 
        
        low_to_high = (
            up_weight_secondary * 
            mortar_projection.secondary_to_mortar_int *
            rho_ad_main
            )
        
        # The density at the interface
        mortar_density = (high_to_low + low_to_high) * lam
        
        # Include the mortar density in the flux towards \Omega_h 
        full_density_flux += (
            mpfa.bound_flux * 
            mortar_projection.mortar_to_primary_int * mortar_density
            )
        
        # The source term for the flux to \Omega_l
        sources_from_mortar = (
            mortar_projection.mortar_to_secondary_int * mortar_density
            )
        
        conservation = div * full_density_flux - sources_from_mortar
        
    else :
        conservation = div * full_density_flux
    # end if
      
    # Construct the conservation equation
    density_wrap = (
        (mass_density.mass * rho_ad_main - 
         mass_density_prev.mass * rho_ad(p_prev)) / dt 
        + conservation
        ) 
    if len(edge_list)>0:
                       
        pressure_trace_from_high = (
            mortar_projection.primary_to_mortar_avg * mpfa.bound_pressure_cell * p
            + (
                mortar_projection.primary_to_mortar_avg * mpfa.bound_pressure_face *
                mortar_projection.mortar_to_primary_int * lam
                )
            )
        
        pressure_from_low = mortar_projection.secondary_to_mortar_avg * p   
        
        robin = pp.ad.RobinCouplingAd(flow_kw, edge_list)
        
        # The interface flux, lambda
        interface_flux = (
            robin.mortar_discr * (
              pressure_trace_from_high - pressure_from_low
                )
            + lam 
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
    
     # For later use, remove the flux on the fracture faces manually
    full_flux = remove_frac_face_flux(full_flux, gb, dof_manager)
    
    # Store the flux 
    data_prev_newton["AD_full_flux"] = full_flux 
    
    if len(edge_list) > 0: 
        data_prev_newton.update({"AD_lam_flux": lam})
    # end if   
    
#%% The Passive tracer
       
    # It is similar to the solute transport equation, which is studied 
    # in detail below, but only applied to one component.
    # The reason I placed the construction of the temperature equation 
    # above the solute transport equation is to avoid having to "redefine"
    # the div, the mortar_projection, etc operators
  
    # Upwind discretization for advection.
    upwind_tracer = pp.ad.UpwindAd(keyword=tracer, grids=grid_list)
    if len(edge_list) > 0:
        upwind_tracer_coupling = pp.ad.UpwindCouplingAd(keyword=tracer, edges=edge_list)
    # end if
    
    # Ad wrapper of boundary conditions
    bc_tracer = pp.ad.BoundaryCondition(keyword=tracer, grids=grid_list)
       
    # For the accumulation term, we need T and teh porosity at  
    # the previous time step. These should be fixed in the Newton-iterations
    if iterate: # Newton-iteration 
        mass_tracer = pp.ad.MassMatrixAd(tracer, grid_list)
        mass_tracer_prev = data_prev_time["mass_tracer_prev"]
        tracer_prev = data_prev_time["tracer_prev"]
    else: # We are interested in "constructing" the equations

        # Mass matrix for accumulation
        mass_tracer = pp.ad.MassMatrixAd(tracer, grid_list)
        mass_tracer_prev = pp.ad.MassMatrixAd(tracer, grid_list) 
       
        # The solute solution at time step n, i.e. 
        # the one we need to use to fine the solution at time step n+1
        tracer_prev = passive_tracer.previous_timestep() 
        
        # Store for Newton iterations
        data_prev_time["tracer_prev"] = tracer_prev
        data_prev_time["mass_tracer_prev"] = mass_tracer_prev    
    # end if-else
    
    # We need four terms for the solute transport equation:
    # 1) Accumulation
    # 2) Advection
    # 3) Boundary condition for inlet
    # 4) boundary condition for outlet.
    abs_full_flux = repeat(full_flux, 1, gb, dof_manager)

    tracer_wrapper = (
        (mass_tracer.mass * passive_tracer - mass_tracer_prev.mass * tracer_prev) / dt
        
        # advection    
        + div * (
            (upwind_tracer.upwind * passive_tracer) * abs_full_flux 
            - upwind_tracer.bound_transport_dir * abs_full_flux * bc_tracer
            - upwind_tracer.bound_transport_neu * bc_tracer
            )
        )
    
    # Add the projections 
    if len(edge_list) > 0:
        
        tracer_wrapper += (
            trace.inv_trace * 
            mortar_projection.mortar_to_primary_int *
            eta_tracer                                    
            )
        
        tracer_wrapper -= (
            mortar_projection.mortar_to_secondary_int *
            eta_tracer 
            )  
    # end if
    
    # Tracer over the interface
    if len(edge_list) > 0:
        
        # Some tools we need
        upwind_tracer_coupling_flux = upwind_tracer_coupling.flux
        upwind_tracer_coupling_primary = upwind_tracer_coupling.upwind_primary
        upwind_tracer_coupling_secondary = upwind_tracer_coupling.upwind_secondary
                
        # First project the concentration from high to low
        trace_of_tracer = trace.trace * passive_tracer
                                                    
        high_to_low_tracer = (
            #upwind_tracer_coupling_flux *
            upwind_tracer_coupling_primary * 
            mortar_projection.primary_to_mortar_avg *  
            trace_of_tracer
            ) 
        
        # Next project concentration from lower onto higher dimension
        low_to_high_tracer = (
            #upwind_tracer_coupling_flux * 
            upwind_tracer_coupling_secondary * 
            mortar_projection.secondary_to_mortar_avg * 
            passive_tracer
            ) 
        
        # Finally we have the transport over the interface equation
        abs_lam = repeat(lam, 1, gb, dof_manager)
        tracer_over_interface_wrapper = (
             upwind_tracer_coupling.mortar_discr * eta_tracer 
               - (high_to_low_tracer + low_to_high_tracer) * abs_lam
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
        
#%% Next, the (solute) transport equations.

    # This consists of terms for accumulation and advection, in addition to boundary conditions.
    # There are two additional complications:
    # 1) We need mappings between the full set of primary species and the advective subset.
    # 2) The primary variable for concentrations is on log-form, while the standard term is
    #    advected.
    # None of these are difficult to handle, it just requires a bit of work.
    
    # The advection problem is set for the aquatic components only.
    # Create mapping between aquatic and all components
    aq_components = data_transport["aqueous_components"]
    cols = np.ravel(
        aq_components.reshape((-1, 1)) + (num_components * np.arange(gb.num_cells() )), order="F"
    )
    sz = num_aq_components * gb.num_cells()
    rows = np.arange(sz)
    matrix_vals = np.ones(sz)
    
    # Mapping from all to aquatic components. The reverse map is achieved by a transpose.
    all_2_aquatic = pp.ad.Matrix(
        sps.coo_matrix(
            (matrix_vals, (rows, cols)),
            shape=(num_aq_components * gb.num_cells(), num_components * gb.num_cells()),
        ).tocsr()
    )
    
    # Upwind discretization for advection.
    upwind = pp.ad.UpwindAd(keyword = transport_kw, grids = grid_list)
    if len(edge_list) > 0:
        upwind_coupling = pp.ad.UpwindCouplingAd(keyword=transport_kw, edges = edge_list)
    # end if
    
    # Ad wrapper of boundary conditions
    bc_c = pp.ad.BoundaryCondition(keyword=transport_kw, grids=grid_list)
    
    # Divergence operator. Acts on fluxes of aquatic components only.
    div = data_grid["divergence_several"]
    
    # Mortar projection between subdomains and mortar grids
    if len(edge_list) > 0 :
        mortar_projection = data_grid["mortar_projection_several"]
    # end if
    
    # Conservation applies to the linear form of the total consentrations.
    # Make a function which carries out both conversion and summation over primary
    # and secondary species.
    def log_to_linear(c: pp.ad.Ad_array) -> pp.ad.Ad_array:
        return pp.ad.exp(c) + cell_S.T * cell_equil_comp * pp.ad.exp(cell_S * c)
    
    # Wrap function in an Ad function, ready to be parsed
    log2lin = pp.ad.Function(log_to_linear, "")
       
    # For the accumulation term, we need T and teh porosity at  
    # the previous time step. These should be fixed in the Newton-iterations
    if iterate: # Newton-iteration 
        mass = pp.ad.MassMatrixAd(mass_kw, grid_list)
        mass_prev = data_prev_time["mass_prev"]
        T_prev = data_prev_time["T_prev"]
    else: # We are interested in "constructing" the equations

        # Mass matrix for accumulation
        mass = pp.ad.MassMatrixAd(mass_kw, grid_list)
        mass_prev = pp.ad.MassMatrixAd(mass_kw, grid_list) 
       
        # The solute solution at time step n, 
        # ie. the one we need to use to fine the solution at time step n+1
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
    
    # NOTE: The upwind discretization matrices are based on the 
    # signs of the fluxes, not the fluxes themselfs.
    # The fluxes are computed above, in the flow part as
    # full_flux = mpfa * p + bound mpfa * p_bound + ...
    # Need to expand the flux vector
    
    expanded_flux = repeat(full_flux, num_aq_components, gb, dof_manager)
   
    transport = (
        (mass.mass * T - mass_prev.mass * T_prev) / dt
        
        # advection    
        + all_2_aquatic.transpose() * div * (
            upwind.upwind * all_2_aquatic * log2lin(log_X) * expanded_flux 
            - upwind.bound_transport_dir * expanded_flux * bc_c 
            - upwind.bound_transport_neu * bc_c
            )
    ) 
    
    # Add the projections 
    if len(edge_list) > 0:
        
        # The trace operator
        trace = data_grid["trace_several"]
        
        # Like above, we need mapping between aquatic and all components,
        # but now with mortar cells
        cols2 = np.ravel(
            aq_components.reshape((-1, 1)) 
            + (num_components * np.arange(gb.num_mortar_cells() )), order="F"
        )
        
        sz2 = num_aq_components * gb.num_mortar_cells()
        rows2 = np.arange(sz2)
        matrix_vals2 = np.ones(sz2)
        
        # Mapping from all to aquatic components. 
        # The reverse map is achieved by a transpose.
        all_2_aquatic_at_interface = pp.ad.Matrix(
            sps.coo_matrix(
                (matrix_vals2, (rows2, cols2)),
                shape=(num_aq_components * gb.num_mortar_cells(), 
                       num_components    * gb.num_mortar_cells() ),
            ).tocsr()
            )
        
        transport += (
            trace.inv_trace * 
            mortar_projection.mortar_to_primary_int *
            all_2_aquatic_at_interface.transpose() * # Behaves like a "coupling factor" to ensure that
            eta                                      # the multiplication is ok  
            )
        
        transport -= (
            mortar_projection.mortar_to_secondary_int *
            all_2_aquatic_at_interface.transpose() *
            eta 
            )  
    # end if
    
    #%% Transport over the interface
    if len(edge_list) > 0:
        
        # Some tools we need
        upwind_coupling_flux = upwind_coupling.flux
        upwind_coupling_primary = upwind_coupling.upwind_primary
        upwind_coupling_secondary = upwind_coupling.upwind_secondary
        
        expanded_lam = repeat(lam, num_aq_components, gb, dof_manager)
        
        # First project the concentration from high to low
        # At the higher-dimensions, we have both fixed 
        # and aqueous concentration. Only the aqueous 
        # concentrations are transported
        trace_of_conc = trace.trace * log2lin(log_X) 
                                                    
        high_to_low_trans = (
            upwind_coupling_flux *
            upwind_coupling_primary * 
            all_2_aquatic_at_interface * 
            mortar_projection.primary_to_mortar_avg *  
            trace_of_conc
            ) 
        
        # Next project concentration from lower onto higher dimension
        # At the lower-dimension, we also have both 
        # fixed and aqueous concentration
        low_to_high_trans = (
            upwind_coupling_flux * 
            upwind_coupling_secondary * 
            all_2_aquatic_at_interface * 
            mortar_projection.secondary_to_mortar_avg * 
            log2lin(log_X) 
            ) 
        
        # Finally we have the transport over the interface equation
        transport_over_interface = (
             upwind_coupling.mortar_discr * eta 
            - (high_to_low_trans + low_to_high_trans) * expanded_lam 
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
        f = (x1 - x2 > 0) 
        f = f.astype(float)
        Fa = sps.diags(f)  # Active
        Fi = sps.diags(1-f) # Inactive 
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
    
    #%% The last step is to feed the equations to the equation manager 
    #   and return the non-linear equations
    equation_manager.equations.clear()
    
    if len(edge_list) > 0:
        equation_manager.equations = {"fluid_conservation": density_wrap,  
                                      "interface_flux": interface_flux,                 
                                      "solute_transport": transport,  
                                      "interface_transport": transport_over_interface, 
                                      "equilibrium": equilibrium_eq,
                                      "dissolution_and_precipitation": mineral_eq,
                                      "passive_tracer": tracer_wrapper,
                                      "tracer_interface_flux": tracer_over_interface_wrapper
                                      }
    else:
        equation_manager.equations = {"fluid_conservation": density_wrap,   
                                      "solute_transport": transport, 
                                      "equilibrium": equilibrium_eq,
                                      "dissolution_and_precipitation": mineral_eq,
                                      "passive_tracer": tracer_wrapper
                                      }
    # end if-else
                    
    return equation_manager