"""
Update various parameters for the computations
"""

import numpy as np
import porepy as pp 
import scipy.sparse as sps

from equations import rho

def equil_constants(gb, temp=None):
    """Calculate the equilibirum constants for a given temperature
    
    """
    
    R = 8.314 # Gas constant
    
    # The enthalpy; 
    # the values are from appendix 2 in Chang and Goldsby
    hco3 =-691.1
    h = 0
    co3 = -676.3
    hso4 = -885.75
    so4 = -907.5
    oh = -229.94
    h2o = -285.8
    caco3 = -1206.9
    ca = -542.96
    caso4 = -1432.69
    
    # The Enthalpy for the reactions
    delta_H = np.array([
        h   + co3 - hco3,
        h   + so4 - hso4,
        h2o - h   - oh  , 
        ca  + co3 - caco3,
        ca  + so4 - caso4 
        ]) * 1e3 # The above values have unit [kJ/mol]
    
    c = delta_H/R
    
    ref_temp = 573.15
    
    if temp is None: # Not a temperature is given, meaning we are in Newton iterations
    
        shape = gb.num_cells() * delta_H.size
        vec = np.zeros(shape)  
        # Idea: Loop over the gb, and fill in cell-wise. Then return as a sparse matrix.
        
        val = 0
        for g,d in gb:
             
            if g.dim == 2:
                data = d
            # end if
            
            # Cell-wise enthaply for the calculations below
            cell_wise_c = np.tile(c, g.num_cells)
        
            # The temperature
            t = d[pp.STATE][pp.ITERATE]["temperature"]
                
            # Expand the temperature for the calculation
            temp = np.repeat(t, repeats=delta_H.shape)
            
            # Temperature factor
            #ref_temp = d[pp.PARAMETERS]["temperature"]["initial_temp"] 
            dt = np.clip(temp-ref_temp, a_min=-ref_temp/6, a_max=ref_temp/6) 
            #temp_factor = dt/(ref_temp*temp)
            
            ref_eq = d[pp.PARAMETERS]["reference"]["equil_consts"] 
            ref_eq = np.tile(ref_eq, g.num_cells)
           
            # Calulate the equilibrium constants using a linearized van't Hoff equation
            taylor_app = ( 
                1 + cell_wise_c * dt / ref_temp**2
         
                )
            cell_equil_consts = ref_eq * taylor_app #* np.exp(cell_wise_c * temp_factor)
            
            # Return
            inds = slice(val, val+g.num_cells*delta_H.size) 
            vec[inds] = cell_equil_consts
            
            # For the next grid
            val += g.num_cells * delta_H.size
        # end g,d-loop
            
        # Splitt the equilibrium constants for components 
        # and precipitation species
        x1 = np.zeros(gb.num_cells() * 3, dtype=int)
        x2 = np.zeros(gb.num_cells() * 2, dtype=int)
        for i in range(gb.num_cells()):
            j = i*5
            inds1 = np.array([j, j+1, j+2]) 
            inds2 = np.array([j+3, j+4])
            
            x1[3*i:3*i+3] = inds1
            x2[2*i:2*i+2] = inds2
        # end i-loop
        
        cell_equil_comp = vec[x1]
        cell_equil_prec = vec[x2]
        
        data["parameters"]["chemistry"].update({
            "cell_equilibrium_constants_comp": sps.diags(cell_equil_comp),
            "cell_equilibrium_constants_prec": sps.diags(cell_equil_prec)
            })
        
        return None
    
    else: # This part is ment to calculate the exponential part 
          # at a given (scalar) temperature  
        dt = np.clip(temp-ref_temp, a_min=-ref_temp/6, a_max=ref_temp/6) 
   
        exponential_part = ( 
          1 + c * dt / ref_temp**2
          )
        return exponential_part
    # end if

def matrix_perm(phi, ref_phi, ref_perm):
    """The matrix permeability
    
    """
    factor1 = np.power((1-ref_phi), 2) / np.power((1-phi), 2)
    factor2 = np.power(phi/ref_phi, 3)
    K = ref_perm * factor1 * factor2 
    return K

def fracture_perm(aperture):
    """Cubic law for the fracture permeability
    
    """
    K = np.power(aperture, 2) / 12 
    return K 

def update_perm(gb):
    """Update the permeability in the matrix and fracture
    
    """
    
    for g,d in gb:     
        
        d_ref = d[pp.PARAMETERS]["reference"]
        if g.dim == gb.dim_max(): # At the matrix
            ref_phi = d_ref["porosity"]
            ref_perm = d_ref["permeability"]
            phi = d[pp.PARAMETERS]["mass"]["porosity"]
            # Matrix permeability
            K = matrix_perm(phi, ref_phi, ref_perm)
            
        else : # In the fracture and intersections
            ref_perm = d_ref["permeability"]
            aperture = d[pp.PARAMETERS]["mass"]["aperture"]
            # Fracture permeability 
            K= fracture_perm(aperture) 
        # end if-else
        
        specific_volume = specific_vol(gb, g)
        dynamic_viscocity = d[pp.PARAMETERS]["flow"]["dynamic_viscosity"]
        
        kxx = K * specific_volume / dynamic_viscocity
        
        d[pp.PARAMETERS]["flow"].update({
            "permeability": kxx,
            "second_order_tensor": pp.SecondOrderTensor(kxx)
            }) 
        
        # We are also interested in the current permeability,
        # compared to the initial one
        d[pp.STATE].update({"ratio_perm": K/ref_perm}) 
        
    # end g,d-loop    
    
def update_mass_weight(gb):
    """Update the porosity-dependent parts in the transport equations:
    For the soltue transport this is the porosity, while for the 
    temeprature equation, it is the heat capacity and conductivity

    
    """
    for g,d in gb:
        
        specific_volume = specific_vol(gb, g)
        
        conc_CaCO3 = d[pp.STATE][pp.ITERATE]["CaCO3"]
        conc_CaSO4 = d[pp.STATE][pp.ITERATE]["CaSO4"]

        # Number of moles
        mol_CaCO3 = conc_CaCO3 * g.cell_volumes 
        mol_CaSO4 = conc_CaSO4 * g.cell_volumes 
        
        # Next convert mol to g
        mass_CaCO3 = mol_CaCO3 * 100.09 * 0.001 # the "100.09" is molar mass, g/mol
        mass_CaSO4 = mol_CaSO4 * 136.15 * 0.001 # the "0.001" from g to kg 
        
        density_CaCO3 = 2.71e3 # kg/m^3
        density_CaSO4 = 2.97e3 # kg/m^3
                                                
        # the mineral volumes                                            
        mineral_vol_CaCO3 = mass_CaCO3 / density_CaCO3 
        mineral_vol_CaSO4 = mass_CaSO4 / density_CaSO4 
        
        # Porosity
        porosity = 1 - (mineral_vol_CaCO3 + mineral_vol_CaSO4) / g.cell_volumes
      
        # Include the non-reactive porosity
        if gb.dim_max() == g.dim:
            porosity -=  0.8
        # end if
        
        d[pp.PARAMETERS]["mass"].update({
            "porosity": porosity,
            "mass_weight": porosity.copy() * specific_volume.copy()
            })
        
        d[pp.PARAMETERS]["flow"].update({
            "mass_weight": porosity.copy() * specific_volume.copy()
            })
        
        d[pp.PARAMETERS]["passive_tracer"].update({
            "mass_weight": porosity.copy() * specific_volume.copy()
            })
        
        # Finally, calulate heat capacity and conduction
        fluid_density = rho(d[pp.STATE][pp.ITERATE]["pressure"], 
                            d[pp.STATE][pp.ITERATE]["temperature"])
        d_temp = d[pp.PARAMETERS]["temperature"]
        
        heat_capacity = (
            porosity.copy() * fluid_density * d_temp["specific_heat_fluid"] +
            (1-porosity.copy()) * d_temp["solid_density"] * d_temp["specific_heat_solid"]
            )
        
        conduction = (
            porosity.copy() * d_temp["fluid_conduction"] + 
            (1-porosity.copy()) * d_temp["solid_conduction"]
            )
        
        d_temp.update({
            "mass_weight": heat_capacity * specific_volume.copy(),
            "second_order_tensor": pp.SecondOrderTensor(conduction * specific_volume.copy()) 
            })
        
    
def update_elliptic_interface(gb):
    """Update the elliptic interfaces, following
    https://github.com/IvarStefansson/A-fully-coupled-numerical-model-of-thermo-hydro-mechanical-processes-and-fracture-contact-mechanics-
    
    """

    for e,d in gb.edges():
        mg = d["mortar_grid"]
        gl, gh = gb.nodes_of_edge(e)
  
        data_l = gb.node_props(gl)
        aperture = data_l[pp.PARAMETERS]["mass"]["aperture"]
        
        Vl = specific_vol(gb, gl)
        Vh = specific_vol(gb, gh) 
        tr = np.abs(gh.cell_faces)
        Vj = mg.primary_to_mortar_int() * tr * Vh 
        
        # The normal diffusivities
        
        # Assume the normal and tangential permeability are equal
        # in the fracture
        ks = data_l[pp.PARAMETERS]["flow"]["permeability"]
        nd = mg.secondary_to_mortar_int() * np.divide(ks, aperture * Vl / 2) * Vj
        d[pp.PARAMETERS]["flow"].update({"normal_diffusivity": nd})    
        
        # Conduction
        fluid_condution = data_l[pp.PARAMETERS]["temperature"]["fluid_conduction"]
        q = 2 * fluid_condution * Vj / (mg.secondary_to_mortar_avg() * aperture) 
        d[pp.PARAMETERS]["temperature"].update({"normal_diffusivity": q})
    # end e,d-loop


def specific_vol(gb,g):
    return gb.node_props(g)[pp.PARAMETERS]["mass"]["specific_volume"]

    
def update_intersection(gb):
    """aperture and specific volume at intersection points (i.e. 0-d)
    
    """
    for g,d in gb:
        
        if g.dim == gb.dim_max():
            open_aperture = d[pp.PARAMETERS]["mass"]["open_aperture"]
        # end if
        #init_volume = d[pp.PARAMETERS]["reference"]["specific_volume"]
        
        
        parent_aperture = []
        num_parent = []
        
        if g.dim < gb.dim_max()-1:
            for edges in gb.edges_of_node(g):
                e = edges[0]
                gh = e[0]
                
                if gh == g:
                    gh = e[1]
                # end if
                
                if gh.dim == gb.dim_max()-1:
                    dh = gb.node_props(gh)
                    a = dh[pp.PARAMETERS]["mass"]["aperture"]
                    ah = np.abs(gh.cell_faces) * a
                    mg = gb.edge_props(e)["mortar_grid"]
                    
                    al = (
                        mg.mortar_to_secondary_avg() *
                        mg.primary_to_mortar_avg() *
                        ah
                        )
                    
                    parent_aperture.append(al)
                    num_parent.append(
                        np.sum(mg.mortar_to_secondary_int().A, axis=1)
                        )
                    # end if
            # end edge-loop
            
            parent_aperture = np.array(parent_aperture)
            num_parent = np.sum(np.array(num_parent), axis=0)
                
            aperture = np.sum(parent_aperture, axis=0) / num_parent 
            
            if pp.PARAMETERS not in d.keys():
                #breakpoint()
                return aperture
            else:
                #breakpoint()
                volume = np.power(aperture, gb.dim_max() - g.dim)
                #volume=np.clip(volume, a_min=1e-5, a_max=1)
                d[pp.PARAMETERS]["mass"].update({"aperture": aperture, 
                                                 "specific_volume": volume})   
                
                d[pp.STATE]["aperture_difference"] = open_aperture - aperture.copy()
            # end if
        # end if

def update_aperture(gb):
    """
    Update the aperture in the fracture (1-D). The 0-D case is handled by update_speific_volume
    s 
    """
    for g,d in gb:
        
        # Only update the aperture in the fracture
        if gb.dim_max()-1 == g.dim :
            ref_param = d[pp.PARAMETERS]["reference"]
         
            # the reactive surface area
            S = ref_param["surface_area"]    
            
            conc_CaCO3 = d[pp.STATE][pp.ITERATE]["CaCO3"]
            conc_CaSO4 = d[pp.STATE][pp.ITERATE]["CaSO4"]

            # Number of moles
            mol_CaCO3 = conc_CaCO3 * g.cell_volumes 
            mol_CaSO4 = conc_CaSO4 * g.cell_volumes 
        
            # Next convert mol to g
            mass_CaCO3 = mol_CaCO3 * 100.09 * 0.001 # the "100.09" is molar mass, g/mol
            mass_CaSO4 = mol_CaSO4 * 136.15 * 0.001
        
            # Mineral densitys    
            density_CaCO3 = 2.71e3 # kg/m^3
            density_CaSO4 = 2.97e3 # kg/m^3
                                                
            # the mineral volumes                                            
            mineral_vol_CaCO3 = mass_CaCO3 / density_CaCO3 
            mineral_vol_CaSO4 = mass_CaSO4 / density_CaSO4 
            
            mineral_width_CaCO3 = np.zeros(S.size)
            mineral_width_CaSO4 = mineral_width_CaCO3.copy()
            
            mineral_width_CaCO3 = mineral_vol_CaCO3 / S
            mineral_width_CaSO4 = mineral_vol_CaSO4 / S
         
            open_aperture = d[pp.PARAMETERS]["mass"]["open_aperture"]
            
            aperture = open_aperture - mineral_width_CaCO3 - mineral_width_CaSO4
            
            # Clip to avoid exactly zero aperture (can cause singular Jacobian)
            aperture = np.clip(aperture, a_min=1e-7, a_max=open_aperture)
            
            d[pp.PARAMETERS]["mass"].update({"aperture": aperture.copy(),
                                             "specific_volume": aperture.copy()})
            
            d[pp.STATE]["aperture_difference"] = open_aperture - aperture.copy()
                        
        # end if
    # end g,d-loop
    
def update(gb):
    """Update the aperture, mass weight and permeability
    
    """
    update_aperture(gb)
    
    # Update at intersection points 
    if gb.dim_min() == gb.dim_max()-2:
        update_intersection(gb)
    # end if
    
    # Update porosity
    update_mass_weight(gb)
    
    # Update permeability
    update_perm(gb)
    
    # Update interface
    update_elliptic_interface(gb)
    
    return gb
   
def update_concentrations(gb, dof_manager, to_iterate=False):
    """Update concenctations
    
    """
    dof_ind = np.cumsum(np.hstack((0, dof_manager.full_dof)))
    x = dof_manager.assemble_variable(from_iterate=True)
    cell_val=0
    for g,d in gb:
        
        if g.dim == gb.dim_max():
            data_chemistry = d[pp.PARAMETERS]["chemistry"]
            eq = data_chemistry["cell_equilibrium_constants_comp"].diagonal() 
        # end if

        for key, val in dof_manager.block_dof.items():
            
            if isinstance(key[0], tuple) is False: # Otherwise, we are at the interface
            
                # The primary species corresponds to the exponential of log_conc,
                # which is stored in the dictionary as "log_X"
                # Moreover, we consider dimension by dimension,
                # thus we also need the "current" dimension of the grid.
                # Lastly, we look at the grids inividually
                
                if g.dim==key[0].dim and g.num_cells==key[0].num_cells:
                    if key[1]=="log_X":
                        inds = slice(dof_ind[val], dof_ind[val+1]) 
                        primary = x[inds]
                # Get the mineral concentrations, in a similar manner
                # (they are needed for computing the change in porosity)
                    elif key[1]=="minerals":
                        inds = slice(dof_ind[val], dof_ind[val+1])
                        mineral = x[inds]
                        
                        break # When we reach this point, we have all the necessary 
                              # information return the concentrations, for the 
                              # particular grid. Hence, we may jump out of the loop 
                      # end if-else
          # end key,val-loop
      
        if not to_iterate:  
            S = data_chemistry["stoic_coeff_S"]   
       
            eq_inds=slice(cell_val, cell_val+ S.shape[0]*g.num_cells)            
            eq_const_on_grid = sps.diags(eq[eq_inds])
            # breakpoint()
            cell_val += g.num_cells * S.shape[0]
            S_on_grid = sps.block_diag([S for i in range(g.num_cells)]).tocsr()
            secondary_species = eq_const_on_grid * np.exp(S_on_grid * primary) 
        # end if
        
        caco3 = mineral[0::2]
        caso4 = mineral[1::2]
        
        if to_iterate:
            d[pp.STATE][pp.ITERATE].update({
                "CaCO3": caco3,
                "CaSO4": caso4
                })
        else:
            d[pp.STATE].update({
            "CaCO3": caco3,
            "CaSO4": caso4,
            
            # Primary
            "Ca2+": np.exp(primary[0::4]),
            "CO3":  np.exp(primary[1::4]),
            "SO4":  np.exp(primary[2::4]),
            "H+":   np.exp(primary[3::4]),
            
            # Secondary
            "HCO3": secondary_species[0::3],
            "HSO4": secondary_species[1::3],
            "OH-" : secondary_species[2::3],   
            })
         
        # end if
          
    # end g,d-loop
