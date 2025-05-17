clear; clc;

function f = system_equations(x)
    x1 = x(1);
    x2 = x(2);
    x3 = x(3);

    f = zeros(3, 1);

    f(1) = 2*x1.^3 + 2*x2.^2 + 3*x3.^2 + 4;
    f(2) = 3*x1 + x2.^3 + 8*x3 + 1000;
    f(3) = 5*x1.^2 + 8*x2 - 7*x3.^2 - 4;
endfunction

function J = eval_jacobian(funcF, x)
    h = 1e-6;
    n = length(x);
    J = zeros(n, n);
    f = funcF(x);

    for j = 1:n
        x_h = x;
        x_h(j) = x(j) + h;
        f_h = funcF(x_h);
        J(:, j) = (f_h - f) / h;
    endfor
endfunction

function bestX = find_initial_guess(funcF, n)
    candidates = [-1.5, -0.5, 0.0, 0.5, 1.5];
    bestNorm = inf; 
    bestX = zeros(n, 1);

    for i = 1:length(candidates)
        val = candidates(i);
        x_candidate = ones(n, 1) * val; 
        f_candidate = funcF(x_candidate);
        norm_val = norm(f_candidate); 

        if norm_val < bestNorm
            bestNorm = norm_val;
            bestX = x_candidate;
        endif
    endfor
endfunction

function [x, f, iter] = newton_method(funcF, x0, eps, maxIter)
    x = x0;
    iter = 0;
    n = length(x);

    while iter < maxIter
        f = funcF(x); 
        current_norm = norm(f);

        printf("iter %d:\n", iter + 1); 
        for i = 1:n
            printf("  x%d = %.5f\n", i, x(i));
        endfor
        printf("  residual = %.5f\n", current_norm);

        if current_norm <= eps
            printf("Convergence reached\n");
            break; 
        endif

        J = eval_jacobian(funcF, x); 

        dX = J \ (-f);

        x = x + dX;

        iter = iter + 1;
    endwhile

    if iter == maxIter && norm(funcF(x)) > eps
        warning("Newtons method did not converge within %d iterations.", maxIter);
    endif

endfunction

eps = 1e-5; 
maxIter = 20;
n_equations = 3;

funcF = @system_equations;

x_initial = find_initial_guess(funcF, n_equations);
f_initial = funcF(x_initial);

printf("Initial guess: ");
printf("%.5f ", x_initial);
printf("\n");
printf("Initial residual norm: %.5f\n", norm(f_initial));

[x_sol, f_final, iterations] = newton_method(funcF, x_initial, eps, maxIter);


final_norm = norm(f_final);
if final_norm <= eps
    printf("Solution found in %d iterations.\n", iterations);
else
    printf("Failed to converge.\n");
endif

printf("\nsolution :\n");
for i = 1:length(x_sol)
    printf("  x%d = %.8f\n", i, x_sol(i));
endfor

printf("\nresidual for each equation:\n");
for i = 1:length(f_final)
    printf("  eq %d: %.5e\n", i, f_final(i));
endfor
printf("final norm: %.5e\n", final_norm);