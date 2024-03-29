"""
Solve a non-linear equation 
"""

import numpy as np
import porepy as pp

from newton import newton_gb
from update_param import update_concentrations
    
def solve(gb, dof_manager, equation_manager):
    """Solve the non-linear equations, and possibly adjust the time step
    
    """
    
    conv, newton_iter, flag = newton_gb(gb, 
                                        equation_manager, 
                                        dof_manager, 
                                        "log_X", 
                                        np.array([-30, 25]) 
                                        )
    
    # See if we have converged. If not need to repeat the current time  
    # with a smaller time step
    gb_2d = gb.grids_of_dimension(gb.dim_max())[0]
    data = gb.node_props(gb_2d) 
    data_transport = data[pp.PARAMETERS]["transport"]
    dt = data_transport["time_step"]
    current_time = data_transport["current_time"]
    
    # If Newton converged, distribute the solution to the
    # next step and use as initial guess at the next time step.
    # Otherwise, repeat the current time step with a smaller time steps
    
    if conv is True and flag == 0:
        current_time += dt
        data[pp.PARAMETERS]["previous_newton_iteration"]["Number_of_Newton_iterations"].append(newton_iter)
        data[pp.PARAMETERS]["previous_time_step"]["time_step"].append(dt)
        
        x = dof_manager.assemble_variable(from_iterate=True)
        dof_manager.distribute_variable(x, to_iterate=False)
        
    elif conv is False or flag == 1:     
        x = dof_manager.assemble_variable(from_iterate=False)
        dof_manager.distribute_variable(x.copy(), to_iterate=True)
    # end if
        
    # Adjust time step, if necessary, while making sure 
    # the time step is not too big nor too small 
    if flag == 0:
        if newton_iter < 5: # Few Newton steps, increase the time step 
            data_transport["time_step"] = np.minimum(dt * 2, 10.0)
    
        elif newton_iter > 22: # Used many Newton iterations 
                               # (or Newton didnt converge in maximum 
                               # number of iterations),  decreace the time step
            data_transport["time_step"] = np.maximum(dt / 4, 1e-15)
        # end if-elif
    else: # flag = 1
        data_transport["time_step"] = np.maximum(dt / 10, 1e-15)
    # end if-else

    # Finally, check that the current time step does not make
    # next time point excced the final time
    
    if flag == 0 and current_time + dt > data_transport["final_time"]:
        dt = data_transport["final_time"] - current_time
        data_transport["time_step"] = dt
    # end if
    
    data_transport["current_time"] = current_time
        
def solve_eqs(gb, dof_manager, equation_manager):
    """
    Solve the non-linear equations

    Parameters
    ----------
    gb : grid bucket
    dof_manager : dof manager
    equation_manager : equation manager

    Returns 
    -------
    gb, dof_manager, equation_manager
    """
    solve(gb, dof_manager, equation_manager)
    update_concentrations(gb, dof_manager)
