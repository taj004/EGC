"""
Main script
@author: uw
"""

# %% Import the neseccary packages
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import porepy as pp

import equations
from solve_non_linear import solve_eqs

import pickle
from pathlib import Path

#%% The grid

def create_mesh(mesh_args):
    """
    Create the computational domain EGC.
    The (physical) domain 2x0.5
    
    Input:
        mesh_args: dict parameters for discretization
        scale: int, scaling parameters for the domain size. It is assumed that
        mesh_args and scale are compatible, i.e. mismatches are the users fault
    
    Return:
        a grid bucket
    """
    pts = np.array([[0.6, 0.2], # End pts 
                    [0.2, 0.8], # Statring pts
                     
                    [0.6, 0.6],
                    [0.2, 0.5],
                    
                    [1.2, 0.6],
                    [0.9, 0.8],
                     
                    [1.7, 0.3],
                    [1.0, 0.2],
                    ]).T
                
    e = np.array([[0,1],[2,3], [4,5], [6,7]]).T
    domain = {"xmin" :0.0, "xmax": 2.0, 
              "ymin" :0.0, "ymax": 1.0}
    network_2d = pp.FractureNetwork2d(pts, e, domain)
    gb = network_2d.mesh(mesh_args)
    
    return gb

# %% Initialize variables related to the chemistry and the domain

# Start with the chemical variables
# like equilibrium constants, stoichiometric coeffients, number of species(aqueous, fixed) etc

# Equilibrium constants. Take exponential, as they are given on log-scale
# Moreover, we consider the reciprocal of the equilibrium constants,
# as we look at reactions
# A_j <-> \sum_i xi_{ij} A_i, and
# A_j = eq_const * prod_i A_{i} ^xi_{ij}
# where A_j are the secondary species, and A_i are components (primary variables).
# To be consistenst with the definition of the equilibirum constant, which is
# eq = prod_i A_i ^xi_{ij} / A_j,
# we thus need to take reciprocal.

equil_consts = 1 / np.exp(np.array([10.339, 1.979, -13.997, -8.406, -4.362])) 
equil_consts[3:5] = 1 / equil_consts[3:5]


# Stochiometric matrix for the reactions between primary and secondary species.
# Between aqueous species
S = sps.csr_matrix(
    np.array(
        [
            [0, 1, 0,  1],
            [0, 0, 1,  1],
            [0, 0, 0, -1],
        ]
    )
)

# Between aqueous and preciptated species
E = sps.csr_matrix(
    np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0]
        ]
    )
)

# The number of primary and secondary species can be inferred from the number of
# columns and rows, respectively, of S_W.
# The indices of which primary species are aqueous are specified below.
# Note that the aqueous secondary species should have no connection with fixed
# primary species (the column of S should be all zeros).

# Bookkeeping
aq_components = np.array([0, 1, 2, 3])
fixed_components = 0
num_aq_components = aq_components.size
num_fixed_components = 0
num_components = num_aq_components + num_fixed_components
num_secondary_components = S.shape[0]

# Define fractures and a gb
dx, dy, dz = 0.12, 0.12, 0.12 

mesh_args = {"mesh_size_frac" : dx,
             "mesh_size_bound": dy,
             "mesh_size_min"  : dz}
gb = create_mesh(mesh_args)

domain = {"xmin": 0, "xmax": gb.bounding_box()[1][0],
          "ymin": 0, "ymax": gb.bounding_box()[1][1]} # domain

# Keywords
mass_kw = "mass"
chemistry_kw = "chemistry"
transport_kw = "transport"
flow_kw = "flow"

pressure = "pressure"
tot_var = "T"
log_var = "log_X"
tracer = "passive_tracer"

minerals = "minerals"
min_1 = "CaCO3"
min_2 = "CaSO4"

mortar_pressure = "mortar_pressure"
mortar_transport = "mortar_transport"
mortar_tracer = "mortar_tracer"

# %% Loop over the gb, and set initial and default data

# Permeabilities
matrix_permeability = 1e-13
interfacial_permeability = 1e2

# Initial uniform pressure
init_pressure = 1000 # Pa;  1e-1 # bar

# Some scaling values to help condition the problem
temp_scale = 1 
scalar_scale = 1
chem_scale = 1
length_scale = 1 

for g, d in gb:
    #print(g.cell_volumes)
    gb.add_node_props(["is_tangential"])
    d["is_tangential"] = True

    # Initialize the primary variable dictionaries
    d[pp.PRIMARY_VARIABLES] = {pressure:    {"cells": 1},
                               tot_var:     {"cells": num_components},
                               log_var:     {"cells": num_components},
                               minerals:    {"cells": 2},
                               tracer:      {"cells": 1} 
                               }

    # Initialize a state
    pp.set_state(d)

    unity = np.ones(g.num_cells)
    
    # --------------------------------- #
    
    # Set inital concentration values
    
    if gb.dim_max() == g.dim:
        
        so4 = 10 * unity
        ca = 0.9 / (equil_consts[4] * so4) * unity
        co3 = 1 / (equil_consts[3] * ca) * unity
        oh = 1.5e3 * unity
        h = equil_consts[2] / oh * unity
        hco3 = equil_consts[0] * h * co3 * unity
        hso4 = equil_consts[1] * h * so4 * unity
    
        # Initally calcite is present, but not anhydrite 
        # But no minerals at the boundary cells
        # (The author experienced some issues at the boundary, e.g. extreme precipitation, 
        # precipitation at outflow bc before the solutes are comming there etc)
        cc = g.cell_centers[0]
        ss = 2 * np.min(cc)
        left = cc < domain["xmin"] + ss
        right = cc > domain["xmax"] - ss
 
        precipitated_init = np.zeros((2,g.num_cells))
        precipitated_init[0] = 20
        precipitated_init[0][left] = 0
        precipitated_init[0][right] = 0
        
        init_X = np.array([ca, co3, so4, h])
        init_X_log = np.log(init_X)
        alpha_init = np.array([hco3, hso4, oh])
        init_T = init_X + S.T * alpha_init + E.T * precipitated_init
        
        # Reshape for the calulations
        init_T = init_T.reshape((init_T.size), order="F")
    else: 
        # Inherit values  
        so4 = so4[0] * unity
        ca = ca[0] * unity
        co3 = co3[0] * unity
        oh = oh[0] * unity
        h = h[0] * unity
        hco3 = hco3[0] * unity
        hso4 = hso4[0] * unity
        
        # Have initally calcite present, but not anhydrite 
        precipitated_init = np.zeros((2, g.num_cells))
        precipitated_init[0] = 20.0
   
        init_X = np.array([ca, co3, so4, h])
        init_X_log = np.log(init_X)
        
        alpha_init = np.array([hco3, hso4, oh])
        init_T = init_X + S.T * alpha_init + E.T * precipitated_init
        init_T = init_T.reshape((init_T.size), order="F")
    # end if-else
    
    # --------------------------------- #
    
    # Initial porosity, permeablity and aperture.
 
    # # Number of moles
    mol_CaCO3 = precipitated_init[0] * g.cell_volumes 
    mol_CaSO4 = precipitated_init[1] * g.cell_volumes

    # Next convert mol to g
    mass_CaCO3 = mol_CaCO3 * 100.09 * 0.001 # the "100.09" is molar mass, g/mol
    mass_CaSO4 = mol_CaSO4 * 136.15 * 0.001 # "0.001" convert g to kg

    # Densitys from
    # https://doi.org/10.1016/B978-0-08-100404-3.00004-4
    density_CaCO3 = 2.71e3  # kg/m^3
    density_CaSO4 = 2.97e3  # kg/m^3

    # the mineral volumes
    mineral_vol_CaCO3 = mass_CaCO3 / density_CaCO3 
    mineral_vol_CaSO4 = mass_CaSO4 / density_CaSO4 
    
    # initial reactive surface area 
    S_0 =  np.power(g.cell_volumes, 2/3)

    mineral_width_CaCO3 = mineral_vol_CaCO3 / S_0
    mineral_width_CaSO4 = mineral_vol_CaSO4 / S_0
    
    open_aperture = 5e-3 
    aperture = open_aperture - (
        mineral_width_CaCO3 +  mineral_width_CaSO4 
        ) 
    
    specific_volume = np.power(aperture, gb.dim_max()-g.dim)
    
    dynamic_viscosity = 1e-3
    
    porosity = (
        1 - (mineral_vol_CaCO3 + mineral_vol_CaSO4) / g.cell_volumes  
        ) 
    
    if g.dim == gb.dim_max():
        porosity -= 0.8
        K = matrix_permeability * unity
    else:
        K = np.power(aperture, 2) / 12
    # end if-else

    # --------------------------------- #

    # Boundary conditions
    bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]

    if bound_faces.size != 0:  # Non-zero in 2D/3D, zero otherwise

        bound_faces_centers = g.face_centers[:, bound_faces]
        
        # Neumann faces
        neu_faces = np.array(["neu"] * bound_faces.size)
        # Define the inflow and outflow in the domain
        inflow = bound_faces_centers[0,:] < domain["xmin"] + 1e-4
        outflow = bound_faces_centers[0,:] > domain["xmax"] - 1e-4
        
        # The BC labels for the flow problem
        labels_for_flow = neu_faces.copy()
        labels_for_flow[outflow] = "dir"
        labels_for_flow[inflow] = "dir"

        # Set the BC values for flow
        bc_values_for_flow = np.zeros(g.num_faces)
        bc_values_for_flow[bound_faces[inflow]] = 7 * init_pressure  
        bc_values_for_flow[bound_faces[outflow]] = init_pressure
        bound_for_flow = pp.BoundaryCondition(g, 
                                              faces=bound_faces, 
                                              cond=labels_for_flow)

        # The boundary conditions for transport. Transport here means
        # transport of solutes
        labels_for_transport = neu_faces.copy()
        labels_for_transport[inflow] = "dir"
        bound_for_transport = pp.BoundaryCondition(g, 
                                                   faces=bound_faces, 
                                                   cond=labels_for_transport)

        # Set bc values. Note, at each face we have num_aq_components
        expanded_left = pp.fvutils.expand_indices_nd(bound_faces[inflow], num_aq_components)
        Ny = bound_faces[inflow].size
        bc_values_for_transport = np.zeros(g.num_faces * num_aq_components)
          
        bc_so4 = so4[0]
        bc_ca = 1 / (equil_consts[4] * bc_so4) # Precipitation of CaSO4
        bc_co3 = 0.5 / (equil_consts[3] * bc_ca)# Dissolution of CaCO3
        bc_oh = oh[0]
        bc_h = equil_consts[2] / bc_oh
        bc_hso4 = equil_consts[1] * bc_h * bc_so4
        bc_hco3 = equil_consts[0] * bc_h * bc_so4
        
        bc_X = np.array([bc_ca, bc_co3, bc_so4, bc_h])
        bc_alpha = equil_consts[0:3] * np.exp(S*np.log(bc_X))
        bc_T = bc_X + S.T * bc_alpha 
        
        bc_values_for_transport[expanded_left] = np.tile(bc_T, Ny)
        
        # Boundary conditions for temperature
        labels_for_temp = neu_faces.copy()
        labels_for_temp[inflow] = "dir"
        labels_for_temp[outflow] = "dir"
        bound_for_temp = pp.BoundaryCondition(g, 
                                              faces=bound_faces, 
                                              cond=labels_for_temp)
        
        # Boundary conditions for the passive tracer
        labels_for_tracer = neu_faces.copy()
        labels_for_tracer[np.logical_or(inflow, outflow)] = "dir"
        bound_for_tracer = pp.BoundaryCondition(g, 
                                                faces=bound_faces, 
                                                cond=labels_for_tracer) 
        
        # The values
        bc_values_for_tracer = np.zeros(g.num_faces)
        bc_values_for_tracer[bound_faces[inflow]] = 1
        
    else:
        # In the lower dimansions, no-flow (zero Neumann) conditions
        bc_values_for_flow = np.zeros(g.num_faces)
        bound_for_flow = pp.BoundaryCondition(g)

        bc_values_for_transport = np.zeros(g.num_faces * num_aq_components)
        bound_for_transport = pp.BoundaryCondition(g)

        bc_values_for_temp = np.zeros(g.num_faces)
        bound_for_temp = pp.BoundaryCondition(g)
        
        bc_values_for_tracer = np.zeros(g.num_faces)
        bound_for_tracer = pp.BoundaryCondition(g)
    # end if-else

    # Initial guess for Darcy flux
    init_darcy_flux = np.zeros(g.num_faces)
    init_darcy_flux[g.get_internal_faces()] = 1.0
    
    # Calulate the effective heat capacity
    fluid_density = equations.rho(init_pressure)
    # --------------------------------- #

    # Set the values in dictionaries
    mass_data = {
        "porosity": porosity.copy(),
        "aperture": aperture.copy(),
        "open_aperture": open_aperture, 
        "initial_aperture": aperture.copy(),
        "specific_volume": specific_volume.copy(),
        "mass_weight": porosity.copy() * specific_volume.copy(),
        "num_components": num_components
    }

    flow_data = {
        "mass_weight": porosity * specific_volume.copy(),
        "bc_values": bc_values_for_flow,
        "bc": bound_for_flow,
        "permeability": K * specific_volume.copy() / dynamic_viscosity,
        "darcy_flux": init_darcy_flux,
        "dynamic_viscosity": dynamic_viscosity
    }

    transport_data = {
        "bc_values": bc_values_for_transport,
        "bc": bound_for_transport,
        "num_components": num_aq_components,
        "darcy_flux": init_darcy_flux
    }
    
    passive_tracer_data = {
        "bc_values": bc_values_for_tracer,
        "bc": bound_for_tracer,
        "darcy_flux": init_darcy_flux,
        "mass_weight": porosity.copy() * specific_volume.copy()
        }

    # The Initial values also serve as reference values
    reference_data = {
        "porosity": porosity.copy(),
        "permeability": K.copy(),
        "permeability_aperture_scaled": K.copy() * specific_volume.copy(),
        "aperture": aperture.copy(),
        "surface_area": S_0,
        "mass_weight": porosity.copy() * specific_volume.copy(),
        "length_scale": length_scale,
        "scalar_scale": scalar_scale,
        "chem_scale": chem_scale,
        "equil_consts": equil_consts
    }

    # --------------------------------- #

    # Set the parameters in the dictionary
    # Treat this different to make sure parameter dictionary is constructed
    d = pp.initialize_data(g, d, mass_kw, mass_data)

    d[pp.PARAMETERS][flow_kw] = flow_data
    d[pp.DISCRETIZATION_MATRICES][flow_kw] = {}
    d[pp.PARAMETERS][transport_kw] = transport_data
    d[pp.DISCRETIZATION_MATRICES][transport_kw] = {}
    d[pp.PARAMETERS][tracer] = passive_tracer_data
    d[pp.DISCRETIZATION_MATRICES][tracer] = {}

    # The reference values
    d[pp.PARAMETERS]["reference"] = reference_data

    # The concentrations
    d[pp.PARAMETERS]["concentration"] = {
        # Primary components
        "Ca2+": init_X[0] ,
        "CO3": init_X[1] ,
        "SO4": init_X[2] ,
        "H+": init_X[3] ,
        
        # Secondary species
        "HCO3": alpha_init[0] ,
        "HSO4": alpha_init[1] ,
        "OH-": alpha_init[2] ,
        
        # Minerals
        "CaCO3": precipitated_init[0] ,
        "CaSO4": precipitated_init[1] ,
        
        "init_SO4": init_X[2][0], 
        "init_OH-": alpha_init[2][0],
    }

    # Iterating data
    d[pp.PARAMETERS]["iterating_param"] = {"porosity": porosity.copy(),
                                           "mass_weight": porosity.copy() * aperture.copy(),
                                           "permeability": K.copy(),
                                           "aperture": aperture.copy()}

    # Set some information only in the highest dimension, 
    # in order to avoid techical issues later on
    if g.dim == gb.dim_max():

        # Make block matrices of S_W and equilibrium constants, one block per cell.
        cell_S = sps.block_diag([S for i in range(gb.num_cells())]).tocsr()
        cell_E = sps.block_diag([E for i in range(gb.num_cells())]).tocsr()     
       
        # Between species
        cell_equil_comp = sps.dia_matrix(
            (np.hstack([equil_consts[0:3] for i in range(gb.num_cells())]), 0),
            shape=(
                num_secondary_components * gb.num_cells(),
                num_secondary_components * gb.num_cells(),
            ),
        ).tocsr()
        
        # Between aquoues and precipitating species
        cell_equil_prec = sps.dia_matrix(
            (np.hstack([equil_consts[3:5] for i in range(gb.num_cells())]), 0),
            shape=(
                E.shape[0] * gb.num_cells(),
                E.shape[0] * gb.num_cells(),
                ),
            ).tocsr()
        
        
        d[pp.PARAMETERS][chemistry_kw] = {
            "equilibrium_constants_comp": equil_consts[0:3],
            "equilibrium_constants_prec": equil_consts[3:5],
            "cell_equilibrium_constants_comp": cell_equil_comp,
            "cell_equilibrium_constants_prec": cell_equil_prec,
            
            "stoic_coeff_S": S,
            "stoic_coeff_E": E,
            "cell_stoic_coeff_S": cell_S,
            "cell_stoic_coeff_E": cell_E
        }

        d[pp.PARAMETERS][transport_kw].update({
            "time_step": 5e-4, 
            "current_time": 0,
            "final_time": 7 * pp.DAY,
            "aqueous_components": aq_components
        })
        d[pp.PARAMETERS]["grid_params"] = {}
        d[pp.PARAMETERS]["previous_time_step"] = {"time_step": []}
        d[pp.PARAMETERS]["previous_newton_iteration"] = {"Number_of_Newton_iterations":[]}
    # end if

    # --------------------------------- #

    # Set initial values

    # First for the concentrations, replicated versions of the equilibrium.
    log_conc = init_X_log.reshape((init_T.shape), order="F")
    
    # Minerals per cell
    mineral_per_cell = precipitated_init.reshape((2 * g.num_cells), order="F")
    
    d[pp.STATE].update({
        "dimension" : g.dim * unity, 
        pressure: init_pressure * unity,
        tot_var: init_T.copy(),
        log_var: log_conc,
        tracer: np.zeros(unity.size),
        
        minerals: mineral_per_cell.copy(),
        min_1: precipitated_init[0].copy() ,
        min_2: precipitated_init[1].copy() ,
        
        "aperture_difference": 0 * unity, 
        "ratio_perm": unity, 

        pp.ITERATE: {
            pressure: init_pressure * unity,
            tot_var: init_T.copy(),
            log_var: log_conc.copy(),
            tracer: np.zeros(unity.size),
            
            minerals: mineral_per_cell.copy(),
            min_1: precipitated_init[0].copy() ,
            min_2: precipitated_init[1].copy() ,
        }
    })
    
    # --------------------------------- #

# end g,d-loop

#pp.plot_grid(gb, "dimension", figsize=(15,12))

#%% Loop over the edges:
    
for e, d in gb.edges():

    # Initialize the primary variables and the state in the dictionary
    d[pp.PRIMARY_VARIABLES] = {mortar_pressure:    {"cells": 1},
                               mortar_transport:   {"cells": num_aq_components},
                               mortar_tracer:      {"cells": 1}
                                }
    pp.set_state(d)

    # Number of mortar cells
    mg_num_cells = d["mortar_grid"].num_cells
    unity = np.ones(mg_num_cells)
    
    # For transport we need the number of aqueos components and inital
    # conditions
    init_X_log_aq = init_X_log[aq_components].T[0] 
    
    # Aqueous part of the total concentration, based on the values from the fractrue(s) 
    init_T_aq = np.exp(init_X_log_aq) + S.T * alpha_init.T[0] 
    init_T_aq = np.tile(init_T_aq, mg_num_cells)
    
    d[pp.STATE].update({
        mortar_pressure: 0.0 * unity,
        mortar_transport: init_T_aq,
        mortar_tracer: np.zeros(unity.size),

        pp.ITERATE: {
            mortar_pressure: 0.0 * unity,
            mortar_transport: init_T_aq.copy(),
            mortar_tracer: np.zeros(unity.size)
        }
    })

    # Set the parameter dictionary
    d = pp.initialize_data(e, d, transport_kw, {"num_components": num_aq_components,
                                                "darcy_flux": unity
                                                })

    # Tracer
    d[pp.PARAMETERS][tracer]={"darcy_flux": unity}
    d[pp.DISCRETIZATION_MATRICES][tracer] = {}
    
    # For flow, we need the the normal diffusivity
    nd = interfacial_permeability * unity
    d[pp.PARAMETERS][flow_kw] = {"normal_diffusivity": nd,
                                  "darcy_flux": unity,
                                  }
    d[pp.DISCRETIZATION_MATRICES][flow_kw] = {}

# end e,d-loop

#%% The data in various dimensions
gb_2d = gb.grids_of_dimension(2)[0]
data_2d = gb.node_props(gb_2d)

gb_1d = gb.grids_of_dimension(1)[0]
data_1d = gb.node_props(gb_1d)

# Fill in grid related parameters
grid_list = [g for g,_ in gb]
edge_list = [e for e,_ in gb.edges()]
data_2d[pp.PARAMETERS]["grid_params"].update({
    "grid_list": grid_list,
    "edge_list": edge_list,
    "mortar_projection_single": pp.ad.MortarProjections(gb, nd=1),
    "mortar_projection_several": pp.ad.MortarProjections(gb, nd=num_components),
    "trace_single": pp.ad.Trace(gb, grid_list, nd=1),
    "trace_several": pp.ad.Trace(gb, grid_list, nd=num_components),
    "divergence_single": pp.ad.Divergence(grid_list, dim=1),
    "divergence_several": pp.ad.Divergence(grid_list, dim=num_aq_components)
    })

#%% Conctruct an dof_manager, equation manager and the initial equations
dof_manager = pp.DofManager(gb)
equation_manager = pp.ad.EquationManager(gb, dof_manager)
equation_manager = equations.gather(gb,
                                    dof_manager=dof_manager,
                                    equation_manager=equation_manager)

#%% Prepere for exporting
to_paraview = pp.Exporter(gb, file_name="vars_to_egc", folder_name="to_study")
time_store = np.array([1., 2., 3., 4., 5., 6., 7.]) * pp.DAY
j=0
fields = ["pressure",
          "Ca2+", "CO3", "SO4", "H+", 
          "HCO3", "HSO4", "OH-",
          "CaCO3", "CaSO4", 
          "passive_tracer",
          "aperture_difference", "ratio_perm"]

current_time = data_2d[pp.PARAMETERS]["transport"]["current_time"]
final_time = data_2d[pp.PARAMETERS]["transport"]["final_time"]

#%% Time loop
while current_time < final_time:

    print(f"Current time {current_time}")

    # Solve
    conv = solve_eqs(gb, dof_manager, equation_manager)
    
    # Connect and collect the equations
    equation_manager = equations.gather(gb,
                                        dof_manager=dof_manager,
                                        equation_manager=equation_manager,
                                        iterate=False)

    current_time = data_2d[pp.PARAMETERS]["transport"]["current_time"]
    
    if j < len(time_store) and np.abs(current_time - time_store[j]) < 10:
        j+=1
        to_paraview.write_vtu(fields, time_step = current_time)
    # end if
# end time-loop

#to_paraview.write_vtu(fields, time_step = final_time)

#%% Store the grid bucket
gb_list = [gb] 
folder_name = "to_study/" # Assume this folder exist
gb_name = "gb_for_egc" 

def write_pickle(obj, path):
    """
    Store the current grid bucket for possible later use
    """
    path = Path(path)
    raw = pickle.dumps(obj)
    with open(path, "wb") as f:
        raw = f.write(raw) 
    return

write_pickle(gb_list, gb_name)

#%% Store time step and number of Newton iterations
newton_steps = data_2d[pp.PARAMETERS]["previous_newton_iteration"]["Number_of_Newton_iterations"]
time_steps = data_2d[pp.PARAMETERS]["previous_time_step"]["time_step"] 

newton_steps = np.asarray(newton_steps)
time_steps = np.asarray(time_steps)
time_points = np.linspace(0, final_time, newton_steps.size, endpoint=True)

vals_to_store = np.column_stack(
    (
      time_points,
      newton_steps,
      time_steps,
      )
    )

file_name = folder_name + "temporal_vals" 
np.savetxt(file_name + ".csv", vals_to_store, 
            delimiter=",", header="time_points, newton_steps, time_steps")

