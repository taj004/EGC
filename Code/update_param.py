"""
Update various parameters for the computations
@author: uw
"""

import numpy as np
import porepy as pp 
import scipy.sparse as sps
from equations import rho

def matrix_perm(phi, ref_phi, ref_perm):
    """
    The matrix permeability
    """
    factor1 = np.power((1-ref_phi), 2) / np.power((1-phi), 2)
    factor2 = np.power(phi/ref_phi, 3)
    K = ref_perm * factor1 * factor2 
    return K

def fracture_perm(aperture):
    """
    Cubic law for the fracture permeability
    """
    K = np.power(aperture, 2) / 12 
    return K 

def update_perm(gb):
    """
    Update the permeability in the matrix and fracture
    """
    
    for g,d in gb:     
        
        d_ref = d[pp.PARAMETERS]["reference"]
        if g.dim == gb.dim_max(): # At the matrix
            ref_phi = d_ref["porosity"]
            ref_perm = d_ref["permeability"]
            phi = d[pp.PARAMETERS]["mass"]["porosity"]
            
            K = matrix_perm(phi, ref_phi, ref_perm)
            
        else : # In the fracture and intersections
            ref_perm = d_ref["permeability"]
            
            aperture = d[pp.PARAMETERS]["mass"]["aperture"]
            length_scale = d_ref["length_scale"]
            
            # Aperture and permeability unscaled w.r.t. length scale 
            aperture_unscaled = aperture * length_scale
            K_unscaled = fracture_perm(aperture_unscaled) 
            
            d[pp.PARAMETERS]["flow"]["perm_unscaled"] = K_unscaled

            K = K_unscaled / np.power(length_scale,2) 

        # end if-else
        
        specific_volume = specific_vol(gb, g)
        dynamic_viscocity = d[pp.PARAMETERS]["flow"]["dynamic_viscosity"]
        
        d[pp.PARAMETERS]["flow"].update({
            "permeability": K * specific_volume * pp.BAR / dynamic_viscocity
            }) 
        
        # We are also interested in the current permeability,
        # compared to the initial one
        d[pp.STATE].update({"ratio_perm": K/ref_perm})
        
    # end g,d-loop    
    
def update_mass_weight(gb):
    """
    Update the mass weights in the transport equations:
    For the soltue transport this is the porosity, while for the 
    temeprature equation, it is the heat capacity
        
    The porosity is updated as "porosity = 1 - sum_m x_m",
    where x_m is the mth volume mineral fraction 
    
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
        
        
        
def update_interface(gb):
    """
    Update the interfacial permeability, following
    https://github.com/IvarStefansson/A-fully-coupled-numerical-model-of-thermo-hydro-mechanical-processes-and-fracture-contact-mechanics-
    """

    for e,d in gb.edges():
        mg = d["mortar_grid"]
        gl, gh = gb.nodes_of_edge(e)
        data_l = gb.node_props(gl)
        aperture = data_l[pp.PARAMETERS]["mass"]["aperture"]
        open_aperture = data_l[pp.PARAMETERS]["mass"]["open_aperture"]
        
        Vl = specific_vol(gb, gl)
        Vh = specific_vol(gb, gh) 
        
        # Assume the normal and tangential permeability are equal
        # in the fracture
        ks = data_l[pp.PARAMETERS]["flow"]["permeability"]
        tr = np.abs(gh.cell_faces)
        Vj = mg.primary_to_mortar_int() * tr * Vh 
        
        # The normal diffusivity
        nd = mg.secondary_to_mortar_int() * np.divide(ks, aperture * Vl / 2) * Vj
        
        d[pp.PARAMETERS]["flow"].update({"normal_diffusivity": nd})    
    # end e,d-loop


def specific_vol(gb,g):
    return gb.node_props(g)[pp.PARAMETERS]["mass"]["specific_volume"]

    
def update_specific_volumes(gb):
    """
    aperture and specific volume at intersection points (i.e. 0-d)
    """
    for g,d in gb:
        
        if g.dim == gb.dim_max():
            open_aperture = d[pp.PARAMETERS]["mass"]["open_aperture"]
        # end if
        
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
                        np.sum(mg.mortar_to_secondary_int().A,axis=1)
                        )
                    # end if
            # end edge-loop
            
            parent_aperture = np.array(parent_aperture)
            num_parent = np.sum(np.array(num_parent), axis=0)
                
            aperture = np.sum(parent_aperture, axis=0) / num_parent 
            volume = np.power(aperture, gb.dim_max() - g.dim)
               
            d[pp.PARAMETERS]["mass"].update({"aperture": aperture, 
                                             "specific_volume": volume})   

            d[pp.STATE]["aperture_difference"] = open_aperture - aperture.copy()             
        # end if
        
    return gb

def update_aperture(gb):
    """
    Update the aperture in the fracture (1-D). The 0-D case is handled by update_speific_volumes 
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
        
            # Densitys from     
            density_CaCO3 = 2.71e3  # g/m^3
            density_CaSO4 = 2.97e3 # g/m^3
                                                
            # the mineral volumes                                            
            mineral_vol_CaCO3 = mass_CaCO3 / density_CaCO3 
            mineral_vol_CaSO4 = mass_CaSO4 / density_CaSO4 
            
            # To ensure we don't divide by zero
            ind = np.where(S>1e-10)[0]
            
            mineral_width_CaCO3 = np.zeros(S.size)
            mineral_width_CaSO4 = mineral_width_CaCO3.copy()
            
            mineral_width_CaCO3[ind] = mineral_vol_CaCO3[ind] / S[ind]
            mineral_width_CaSO4[ind] = mineral_vol_CaSO4[ind] / S[ind]
            #breakpoint()
            #initial_aperture = d[pp.PARAMETERS]["mass"]["initial_aperture"]
            open_aperture = d[pp.PARAMETERS]["mass"]["open_aperture"]
            
            aperture = open_aperture - mineral_width_CaCO3 - mineral_width_CaSO4
            
            # Clip to avoid exactly zero aperture
            aperture = np.clip(aperture, a_min=1e-5, a_max=open_aperture)
            
            d[pp.PARAMETERS]["mass"].update({"aperture": aperture.copy(),
                                             "specific_volume": aperture.copy()})
            
            d[pp.STATE]["aperture_difference"] = open_aperture - aperture.copy()
                        
        # end if
    # end g,d-loop
    
def update(gb):
    """
    Update the aperture, mass weight and permeability
    """
    update_aperture(gb)
    
    # Update at intersection points 
    if gb.dim_min() == gb.dim_max()-2:
        update_specific_volumes(gb)
        
    # Update porosity
    update_mass_weight(gb)
    
    # Update permeability
    update_perm(gb)
    
    # Update interface
    update_interface(gb)
    
    return gb
   
def update_concentrations(gb, dof_manager, to_iterate=False):
    """
    Update concenctations
    """
    dof_ind = np.cumsum(np.hstack((0, dof_manager.full_dof)))
    x = dof_manager.assemble_variable(from_iterate=True)
    val=0
    for g,d in gb:
        
        if g.dim == gb.dim_max():
            data_chemistry = d[pp.PARAMETERS]["chemistry"]
            eq = data_chemistry["cell_equilibrium_constants_comp"] 
        # end if

        for key, val in dof_manager.block_dof.items():
            
            if isinstance(key[0], tuple) is False: # Otherwise, we are at the interface
            
                # The primary species corresponds to the exponential of log_conc,
                # which is stored in the dictionary as "log_X"
                # Moreover, we consider dimension by dimension,
                # thus we also need the "current" dimension of the grid.
                # Lastly, we look at the grids inividually
                
                if key[1]=="log_X" and g.dim==key[0].dim and g.num_cells==key[0].num_cells:
                        inds = slice(dof_ind[val], dof_ind[val+1]) 
                        primary = x[inds]
                # Get the mineral concentrations, in a similar manner
                # (they are needed for computing the change in porosity)
                elif key[1]=="minerals" and g.dim==key[0].dim and g.num_cells==key[0].num_cells:
                        inds = slice(dof_ind[val], dof_ind[val+1])
                        mineral = x[inds]
                        
                        break # When we reach this point, we have all the necessary 
                              # information return the concentrations, for the 
                              # particular grid. Hence, we may jump out of the loop 
                # end if-else
          # end key,val-loop
          
        S = data_chemistry["stoic_coeff_S"]   
        S_on_grid = sps.block_diag([S for i in range(g.num_cells)]).tocsr()
        
        eq_inds=slice(val, val+ g.num_cells)
        eq_const_on_grid = sps.dia_matrix(
                (np.hstack([eq.diagonal()[eq_inds] for i in range(g.num_cells)]), 0),
                shape=(
                    S.shape[0] * g.num_cells,
                    S.shape[0] * g.num_cells,
                    ),
                ).tocsr() 
        
        val += g.num_cells
        secondary_species = eq_const_on_grid * np.exp(S_on_grid * primary) 
        
        caco3 = mineral[0::2]
        caso4 = mineral[1::2]
        
        if to_iterate:
            d[pp.STATE][pp.ITERATE].update({
                "CaCO3": caco3,
                "CaSO4": caso4
                })
            
            d[pp.PARAMETERS]["iterating_param"].update({ 
            # Primary
            "Ca2+": np.exp(primary[0::4]),
            "CO3":  np.exp(primary[1::4]),
            "SO4":  np.exp(primary[2::4]),
            "H+":   np.exp(primary[3::4]),
            
            # Secondary
            "HCO3": secondary_species[0::3],
            "HSO4": secondary_species[1::3],
            "OH-" : secondary_species[2::3],   
            
            # Minerals
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
        
            d[pp.PARAMETERS]["concentration"].update({ 
            # Primary
            "Ca2+": np.exp(primary[0::4]),
            "CO3":  np.exp(primary[1::4]),
            "SO4":  np.exp(primary[2::4]),
            "H+":   np.exp(primary[3::4]),
            
            # Secondary
            "HCO3": secondary_species[0::3],
            "HSO4": secondary_species[1::3],
            "OH-" : secondary_species[2::3],   
            
            # Minerals
            "CaCO3": caco3,
            "CaSO4": caso4
                })              
        # end if
          
    # end g,d-loop
