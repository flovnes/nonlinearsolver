using MathNet.Numerics.LinearAlgebra;
using NCalc;

public class Lab3 {
    delegate double Function(Vector<double> x);
    
    static Function ParseEquation(string equation) {
        return x => {
            var expr = new Expression(equation);
            for (int i = 0; i < x.Count; i++)
                expr.Parameters[$"x{i + 1}"] = x[i];
            expr.EvaluateFunction += delegate (string name, FunctionArgs args) {
                switch(name.ToLower()) {
                    case "sin":
                        args.Result = Math.Sin(Convert.ToDouble(args.Parameters[0].Evaluate()));
                        break;
                    case "cos":
                        args.Result = Math.Cos(Convert.ToDouble(args.Parameters[0].Evaluate()));
                        break;
                    case "tan":
                    case "tg":
                        args.Result = Math.Cos(Convert.ToDouble(args.Parameters[0].Evaluate()));
                        break;
                    default: break;
                }
            };
            return Convert.ToDouble(expr.Evaluate());
        };
    }

    static Vector<double> EvalF(Function[] equations, Vector<double> x) {
        var f = Vector<double>.Build.Dense(equations.Length);
        for (int i = 0; i < equations.Length; i++)
            f[i] = equations[i](x);
        return f;
    }

    static Vector<double> InitialGuess(Function[] equations) {
        int n = equations.Length;
        var candidates = new[] { -1.5, -0.5, 0.0, 0.5, 1.5 };
        var bestX = Vector<double>.Build.Dense(n);
        var bestNorm = double.MaxValue;

        foreach (var val in candidates) {
            var x = Vector<double>.Build.Dense(n, val);
            var f = EvalF(equations, x);
            var norm = f.L2Norm();
            if (norm < bestNorm) {
                bestNorm = norm;
                bestX = x;
            }
        }
        return bestX;
    }

    static Matrix<double> EvalJ(Function[] equations, Vector<double> x) {
        double h = 1e-6;
        int n = equations.Length;
        var J = Matrix<double>.Build.Dense(n, n);
        var x_h = x.Clone();

        for (int j = 0; j < n; j++) {
            x_h[j] = x[j] + h;
            var f_h = EvalF(equations, x_h);
            var f = EvalF(equations, x);
            
            for (int i = 0; i < n; i++)
                J[i, j] = (f_h[i] - f[i]) / h;
            
            x_h[j] = x[j];
        }
        return J;
    }
    
    static void Main() {
        double eps = 1e-5;
        int maxIter = 20;

        Console.WriteLine("Example:\n2*Pow(x1,3) + 2*Pow(x2,2) + 3*Pow(x3,2) + 4\n3*x1 + Pow(x2,3) + 8*x3 + 1000\n5*Pow(x1,2) + 8*x2 - 7*Pow(x3,2) - 4\n");
        Console.WriteLine("Press Enter to use the example system, type a number of equations N to write a custom system:");
        string input = Console.ReadLine();

        // 1. Read and parse equations
        ReadEquations(input, out Function[] equations);

        // 2. Eval initial guess with one of known easy methods
        var x = InitialGuess(equations);
        var f = EvalF(equations, x);

        Console.WriteLine($"initial guess: {string.Join(", ", x)}");
        Console.WriteLine($"initial residual: {f.L2Norm()}");

        int iter = NewtonMethod(eps, maxIter, equations, ref x, ref f);

        // 3. calculate residuals
        if (f.L2Norm() <= eps)
            Console.WriteLine($"solution found in {iter} iterations");
        else
            Console.WriteLine("failed to converge");

        // 4. перевірка
        Console.WriteLine("residual for each equation:");
        for (int i = 0; i < f.Count; i++)
            Console.WriteLine($"eq {i + 1}: {f[i]:F5}");
    }

    private static int NewtonMethod(double eps, int maxIter, Function[] equations, ref Vector<double> X, ref Vector<double> f) {
        int iter = 0;
        while (f.L2Norm() > eps && iter < maxIter) {
            // 1. eval F
            f = EvalF(equations, X);

            // 2. eval -(J^-1)
            var J = EvalJ(equations, X);
            var Jinv = J.Inverse();

            // 3. deltaP = -J^-1 * F
            var dP = -Jinv * f;
            X += dP;

            iter++;
            Console.WriteLine($"iter {iter}:");
            for (int i = 0; i < X.Count; i++)
                Console.WriteLine($"x{i + 1} = {X[i]:F5}");
            Console.WriteLine($"residual = {f.L2Norm():F5}");
        }

        return iter;
    }

    private static void ReadEquations(string input, out Function[] equations) {
        if (input == "") {
            equations = [
                ParseEquation("2*Pow(x1,3) + 2*Pow(x2,2) + 3*Pow(x3,2) + 4"),
                ParseEquation("3*x1 + Pow(x2,3) + 8*x3 + 1000"),
                ParseEquation("5*Pow(x1,2) + 8*x2 - 7*Pow(x3,2) - 4"),
            ];
        } else {
            int n = int.Parse(input);
            Console.WriteLine($"Enter {n} equations:");
            equations = new Function[n];
            for (int i = 0; i < n; i++)
                equations[i] = ParseEquation(Console.ReadLine());
        }
    }
}