#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solve a system of non-linear equations on a GB,
using Newton's method and the AD-framework in PorePy

@author: uw
"""

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import porepy as pp

import equations
import update_param

def update_darcy(gb, dof_manager):
    """
    Update the darcy flux in the parameter dictionaries 
    """
    
    # Get the Ad-fluxes
    gb_2d = gb.grids_of_dimension(gb.dim_max())[0]
    data = gb.node_props(gb_2d)
    full_flux = data[pp.PARAMETERS]["previous_newton_iteration"]["AD_full_flux"]
    
    # Convert to numerical values
    num_flux = pp.ad.Expression(full_flux, dof_manager).to_ad(gb)
    if hasattr(num_flux, "val"):
        num_flux = num_flux.val
    # end if
    
    # Get the signs
    sign_flux = np.sign(num_flux)
    
    # Finally, loop over the gb and return the signs of the darcy fluxes
    val = 0
    for g,d in gb:
        inds = slice(val, val + g.num_faces) 
        x = sign_flux[inds]
        # For the solute transport we only need the signs
        d[pp.PARAMETERS]["transport"]["darcy_flux"] = x.copy()
        d[pp.PARAMETERS]["passive_tracer"]["darcy_flux"] = x.copy()
        
        # For the flow, we need the absolute value of the signs
        d[pp.PARAMETERS]["flow"]["darcy_flux"] = np.abs(x.copy())
        
        val += g.num_faces
    # end g,d-loop
    
    #Do the same over the interfaces 
    if gb.dim_max() > gb.dim_min():
        
        # The flux
        edge_flux = data[pp.PARAMETERS]["previous_newton_iteration"]["AD_lam_flux"]
        num_edge_flux = pp.ad.Expression(edge_flux, dof_manager).to_ad(gb).val
        sign_edge_flux = np.sign(num_edge_flux) 
        
        val = 0
        for e,d in gb.edges():
            nc = d["mortar_grid"].num_cells
            inds = slice(val, val+nc)
            x = sign_edge_flux[inds]
            d[pp.PARAMETERS]["transport"]["darcy_flux"] = x.copy()
            d[pp.PARAMETERS]["flow"]["darcy_flux"] = x.copy()
            d[pp.PARAMETERS]["passive_tracer"]["darcy_flux"] = x.copy()
            val += nc
        # end e,d-loop

    return

def clip_variable(x, dof_manager, target_name, min_val, max_val):
    """
    Helper method to cut the values of a target variable.
    Intended use is the concentration variable.
    """
    dof_ind = np.cumsum(np.hstack((0, dof_manager.full_dof)))
    
    for key, val in dof_manager.block_dof.items():
        if key[1] == target_name:
            inds = slice(dof_ind[val], dof_ind[val + 1])
            x[inds] = np.clip(x[inds], a_min=min_val, a_max=max_val)
        # end if
    # end key,val-loop
  
    return x

def backtrack(equation, dof_manager, 
              grad_f, p_k, x_k, f_0, 
              maxiter=5, min_tol=1e-3):
    """
    Compute a stp size, using Armijo interpolation backtracing
    """
    # Initialize variables
    c_1 = 1e-4
    alpha = 1.0 # initial step size
    dot = grad_f.dot(p_k)
    
    flag=0
    
    # The function to be minimized    
    def phi(alpha):
        dof_manager.distribute_variable(x_k + alpha * p_k, to_iterate=True)
        F_new = equation.assemble_rhs()
        phi_step = 0.5*F_new.dot(F_new)
        return phi_step
    
    # Variables
    f_k = f_0 
    alpha_prev = 1.0 # Initial step size
    phi_old = phi(alpha) 
    phi_new = phi_old.copy()
    
    for i in range(maxiter):
        
        # If the first Wolfe condition is satisfied, we stop
        f_new = phi_new.copy()
        if f_new < f_k + alpha*c_1*dot:
            #print(alpha)
            break
        # end if
    
        # Upper and lower bounds
        u = 0.5 * alpha
        l = 0.1 * alpha
    
        # Compute the new step size
        if i == 0: # remember that we have not updated the iteration index yet, 
                    # hence we use the index one lower than what we expect
            
            # The new step size                                     
            numerator = 2 * (phi_old - f_k - dot)
            alpha_temp = -dot/numerator
        else: 
            # The matrix-vector multiplication. 
            mat = np.array([ [ 1/alpha**2, -1/alpha**2 ],
                             [-alpha_prev/alpha**2, alpha/alpha_prev**2 ] ] )
            
            vec = np.array([ phi_old - f_k - dot*alpha,
                             phi_new - f_k - dot*alpha_prev ] )
            
            numerator = alpha - alpha_prev 
        
            a,b = (1/ numerator) * np.matmul(mat, vec)
            
            if np.abs(a)>1e8 or np.abs(b)>1e8:
                flag=1
                break
            # end if
            
            if np.abs(a) < 1e-3: # cubic interpolation becomes quadratic interpolation
                alpha_temp = -dot/(2*b)
            else:
                alpha_temp = (-b + np.sqrt(np.abs(b**2 - 3*a*dot))) / (3*a)
            # end if-else
       # end if-else    
       
        # Check if the new step size is to big
        # From a safty point of view, this helps if alpha_temp is inf.
        # Is it ok to use if alpha_temp is nan?
        alpha_temp = min(alpha_temp,u)
    
        # Update the values, while ensuring that step size is not too small
        alpha_prev = alpha
        phi_old = phi_new  
        alpha = max(alpha_temp, l)
        phi_new = phi(alpha) 
        
        # Check if norm(alpha*p_k) is small. Stop if yes.
        # In such a case we might expect convergence
        if np.linalg.norm(alpha*p_k) < min_tol:
            dof_manager.distribute_variable(x_k+alpha*p_k, to_iterate=True)
            break
        # end if
        
    # end i-loop
        
    return flag

def backtrack_2(equation, dof_manager,
                grad_f, p_k, x_k, f_0,
                maxiter=5, min_tol=1e-3):
    """
    Compute a stp size, using backtracing

    """
    # Initialize variables
    c_1 = 1e-4
    alpha = 1.0 # initial step size
    dot = grad_f.dot(p_k)

    # The function to be minimized    
    def phi(alpha):
        dof_manager.distribute_variable(x_k + alpha*p_k, to_iterate=True)
        F_new = equation.assemble_rhs()
        phi_step = 0.5*F_new.dot(F_new)
        return phi_step
    
    # Variables
    f_k = f_0     
    rho = 0.5
    phi_old = phi(alpha) 
    phi_new = phi_old.copy()
    
    for i in range(maxiter):
        
        # If the first Wolfe condition is satisfied, we stop
        f_new = phi_new.copy()
        if f_new < f_k + alpha*c_1*dot:
            break
        # end if
        
        # Upper and lower bounds
        u = 0.5 * alpha
        l = 0.1 * alpha
        
        # Compute the new step size
        alpha *= rho
        alpha = min(u, max(alpha, l)) 
        phi_new = phi(alpha)
        
        # Check if norm(alpha*p_k) is small. Stop if yes.
        # In such a case we might expect convergence
        if  np.linalg.norm(alpha*p_k) < min_tol:
            break
        # end if
        
    # end i-loop
        
    return 

#@profile
def cond_est(R):
    """
    Estimate the L1-condtion number of a matrix, using the R-factor from the QR-factorization    
    
    Parameters
    ----------
    R: Upper triangular matrix
    
    Return
    ----------
    est: An estimate of the condition number
    """    
    # Get the size on R. It should be nxn
    n = R.shape[0]
    
    # Initialize variables
    x, p = np.zeros(n), np.zeros(n)
    pm = np.zeros(n)
    est = R[0,0]
    
    for j in range(1,n):     
        temp = np.abs(R[j, j]) + np.abs(R[0:j-1, j]).sum()
        est = np.maximum(temp, est)        
    # end j-loop
    
    x[0] = 1/est 
    
    i = np.arange(1,n)
    p[i] = R[0,i]*x[0]
    
    
    # Main loop
    for j in range(1,n):
        # Select ej and compute xj
        xp = (1-p[j]) / R[j,j]
        xm = (-1-p[j]) / R[j,j]
        temp_xp = np.abs(xp)
        temp_xm = np.abs(xm)
        
        i = np.arange(j+1,n)
        abs_R = np.abs(R[i,i])
        pm[i] = p[i] + R[j,i]*xm
        temp_xm += np.sum(np.abs(pm[i]) / abs_R) 
        p[i] = p[i] + R[j,i]*xp
        temp_xp += np.sum(np.abs(p[i]) / abs_R)
        
        if temp_xp >= temp_xm : # ej=1
            x[j] = xp 
        else :                  # ej=-1
            x[j] = xm
            p[i] = pm[i]
        # end if-else
        
    # end j-loop
    
    xnorm = np.abs(x).sum()
    est = est/xnorm
    
    # Solve a linear system
    x = backsubstitution(R, x)
    
    xnorm = np.abs(x).sum()
    est = est*xnorm        
    
    return est

def backsubstitution(R, b):
    """
    Backsubstitution to solve a linear trianglar system of equations
    """
    
    n = b.size    
    x = np.zeros(n)
    
    for i in range(n-1, -1,-1):
        j = np.arange(i+1, n)
        b[i] -= np.dot(R[i,j], x[j])
        x[i] = b[i] / R[i,i]
    # end i-loop
    
    return x

def perturbed_Jacobi(J):
    """
    Perturb the Jacobian matrix, if it is ill-conditioned
    """
    m,n = J.shape
    
    if m != n:
        raise ValueError("Jacobian is not square")
    # end if
    
    I = sps.eye(m) 
    J_dot_J = J.T.dot(J)
    
    Htemp = J_dot_J.A
    Hnorm = Htemp[0,].sum()
    
    for i in range(1,n):
        j1, j2 = slice(1,i), slice(i+1,n)
        temp = Htemp[j1,i].sum() + Htemp[i,j2].sum() 
        Hnorm = max(Hnorm, temp)
    # end i-loop
    
    H = J_dot_J + np.sqrt(n * 2.2e-16) * Hnorm * I 
    
    return H

def scaling_matrix(dof_manager):
    """
    Create a diagonal scaling matrix 
    """
    
    dof_ind = np.cumsum(np.hstack((0, dof_manager.full_dof)))
    # We are primarily interesed in scaling the temperature part,
    # Put 1 otherwis
    D = np.ones(dof_manager.num_dofs())
    
    # for key, val in dof_manager.block_dof.items():
    #     inds = slice(dof_ind[val], dof_ind[val+1])
    #     if key[1]=="pressure":
    #         D[inds] = 1 / 100
    # # end key, val-loop
    
    D = sps.diags(D, 
                  shape=(dof_manager.num_dofs(),dof_manager.num_dofs()), 
                  format="csr")
    return D
    
def newton_gb(gb: pp.GridBucket, 
              equation: pp.ad.EquationManager,
              dof_manager: pp.DofManager, 
              target_name: str = "",
              clip_low_and_up: np.array = np.array([1e-100, 1e100])):
    """
    Newton's method applied to an equation, pulling values and state from gb.
    
    Parameters
    ----------
    equation, The equation we want to solve
    dof_manager, the associated dof_manager
    target_name, string. Value used to target a keyword and clip the associated numerical values 
    clip_low_and_up, numpy array. the upper and lower bound for clipping. 
          The form is np.array([low, up]). The values are interpreted as log values, 
          e.g. np.array([-30,25]), where -30 and 25 is interpreted as
              -30=log(x1), 25=log(x2)     
    
    Returns
    ----------
    conv, bool, whether Newton converged within a tolerance and maximum number of iterations
    i, int, the number of iterations used
    """
  
    J, resid = equation.assemble_matrix_rhs()
    norm_orig = np.linalg.norm(resid)
    print(norm_orig)
    
    # The upper and lower bound
    min_val = clip_low_and_up[0]
    max_val = clip_low_and_up[1]
      
    # Check if the Jacobian is badly conditioned
    # R = np.linalg.qr(J.A, mode="r") 

    # if True in (np.abs(R.diagonal()) < 1e-8) :
    #     ill_cond = True
    #     est = 0
    # else :
    #     ill_cond = False
    #     est = cond_est(R)  
    # # end if-else 
    
    ill_cond=False
    est=1e2
    
    conv = False
    i = 0
    maxit = 25
    
    flag = 0
      
    # Scaling matrix
    D = scaling_matrix(dof_manager)
    
    while conv is False and i < maxit:
        
        # Compute the search direction
        grad_f = J.T.dot(-resid)
      
        if ill_cond or est > 1e10: 
            H = perturbed_Jacobi(J)
            b = -grad_f
            DH=D*H
            Db=D*b
            dx = spla.spsolve(DH,Db, use_umfpack=False)            
        else:
            DJ = D*J
            Dr = D*resid
            dx = spla.spsolve(DJ, Dr, use_umfpack=False) 
        # end if-else
       
        # Solution from prevous iteration step
        x_prev = dof_manager.assemble_variable(from_iterate=True)
        f_0 = 0.5 * resid.dot(resid)
        
        # Step size
        flag=backtrack(equation, dof_manager, grad_f, dx, x_prev, f_0)
        
        # New solution
        x_new = dof_manager.assemble_variable(from_iterate=True)
        
        if np.any(np.isnan(x_new)):
            flag = 1
        # end if
        
        if flag==1:
            break
        
        x_new = clip_variable(x_new.copy(), dof_manager, target_name, 
                              min_val, max_val) 
        
        # x_new = clip_variable(x_new.copy(), dof_manager, "minerals", 
        #                       np.exp(min_val), np.exp(max_val) )     
        
        dof_manager.distribute_variable(x_new.copy(), to_iterate=True)
        
        # --------------------- #
        
        
        # Update the Darcy flux in the parameter dictionries
        update_darcy(gb, dof_manager)   
        
        # Update concentrations in the dictionary
        update_param.update_concentrations(gb, dof_manager, to_iterate=True)
        
        # Update the grid parameters
        update_param.update(gb)
        
        # --------------------- # 
        
        # Increase number of steps
        i += 1
        
        # Update equations
        equation = equations.gather(gb, 
                                    dof_manager=dof_manager,
                                    equation_manager=equation,
                                    iterate=True
                                    ) 
        J, resid = equation.assemble_matrix_rhs()
      
        # Measure the error    
        norm_now = np.linalg.norm(resid)
        err_dist = np.linalg.norm(dx, np.inf) 
        # Stop if converged. 
        if norm_now < 1e-4 * norm_orig or err_dist < 1e-5 * np.linalg.norm(x_new, np.inf):
            print("Solution reached")
            conv = True
        # end if
            
    # end while
    
    # Print some information
    if flag == 0:
        print(f"Number of Newton iterations {i}")
        print(f"Residual reduction {norm_now / norm_orig}")
    elif flag ==1:
        print("Warning: NaN detected")
    # Return the number of Newton iterations; 
    # we use it to adjust the time step
    return conv, i, flag 
    
# end newton
