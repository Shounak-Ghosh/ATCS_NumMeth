import java.io.*;

/**
 * Explores the issues encountered when computing the outputs of the quadratic formula for extreme values.
 * Sin function exploration is incomplete.
 *
 * @author Shounak Ghosh
 * @version 1/24/22
 */
public class QuadraticExploration
{
    private static final double LARGE_VALUE = 100000.0;
    private static final double VALUE = 100;

    /**
     * Driver Method
     *
     * @param args Command-line arguments
     */
    public static void main(String[] args)
    {
        // Read in values for a, b, c
        double a = 1.0, b = 4.0, c = 4.0;
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        try
        {
            System.out.print("a: ");
            a = Double.parseDouble(br.readLine());
            System.out.print("b: ");
            b = Double.parseDouble(br.readLine());
            System.out.print("c: ");
            c = Double.parseDouble(br.readLine());
        }
        catch (IOException ioe)
        {
            System.out.println(ioe);
        }

        // Calculates the roots based on the quadratic formula
        double r1 = (-b + Math.sqrt(b * b - 4 * a * c)) / (2 * a);
        double r2 = (-b - Math.sqrt(b * b - 4 * a * c)) / (2 * a);

        System.out.println("Solutions to " + a + "x^2 + " + b + "x + " + c + " are: " + r1 + " and " + r2);

        // Explore what happens when b^2 >> 4ac. Find the b^2/4ac ratio that causes problems.
        // Use a = 1, c = 1, and some large value of b (ex. 4720372234838).
        // r1 will evaluate to 0 (which is incorrect because f(0) = 1), resulting in the need for the more careful computation below


        // Smaller root calculation via formula: r1 = c/(a*r2), found by algebraic manipulation of the quadratic formula
        if (b > 0)
        {
            r1 = c / (a * r2);
            System.out.println("Smaller root calculation: " + r1);
        } else
        {
            r2 = c / (a * r1);
            System.out.println("Smaller root calculation: " + r2);
        }


        System.out.println("Verifying zeros: f(" + r1 + ") = " + quadraticEvaluation(a, b, c, r1) + ", f(" + r2 + ") = " + quadraticEvaluation(a, b, c, r2));

        // Evaluating f1 and f2 at large positive and negative values
        System.out.println("\nf1(" + -LARGE_VALUE + ") = " + f1(-LARGE_VALUE) + ", f1(" + LARGE_VALUE + ") = " + f1(LARGE_VALUE));
        System.out.println("f2(" + -LARGE_VALUE + ") = " + f2(-LARGE_VALUE) + ", f2(" + LARGE_VALUE + ") = " + f2(LARGE_VALUE) + "\n");

        // Evaluating f(x,delta) at x = 100 and various delta
        double[] delta = new double[]{1.0, 0.1, 0.01, 0.001, 0.0001, 1E-5, 1E-6, 1E-7, 1E-8};

        System.out.println("f(x, delta) = sin(x + delta) - sin(x)\n");
        for (int i = 0; i < delta.length; i++)
        {
            System.out.println("f(" + VALUE + ", " + delta[i] + ") = " + sinFunction(VALUE, delta[i]));
        }

        System.out.println("\nf(x, delta) with sine subtraction identity\n");
        for (int i = 0; i < delta.length; i++)
        {
            System.out.println("f(" + VALUE + ", " + delta[i] + ") = " + betterSinFunction(VALUE, delta[i]));
        }

    }

    /**
     * Evaluates a quadratic
     *
     * @param a Constant on the x^2 term
     * @param b Constant on the x term
     * @param c Constant on the x^0 term
     * @param x The input value
     * @return The output of the function
     */
    private static double quadraticEvaluation(double a, double b, double c, double x)
    {
        return a * x * x + b * x + c;
    }

    /**
     * Calculates e^x / (e^x - 1)
     *
     * @param x The input value
     * @return The double output of the function
     */
    private static double f1(double x)
    {
        return Math.pow(Math.E, x) / (Math.pow(Math.E, x) - 1);
    }

    /**
     * Calculates 1 / (1-e^-x)
     *
     * @param x The input value
     * @return The double output value of the function
     */
    private static double f2(double x)
    {
        return 1 / (1 - Math.pow(Math.E, -x));
    }

    /**
     * Calculates sin(x + delta) - sin(x)
     *
     * @param x     The input value
     * @param delta The variance
     * @return The double output of the computation
     */
    private static double sinFunction(double x, double delta)
    {
        return Math.sin(x + delta) - Math.sin(x);
    }

    /**
     * Calculates sin(x + delta) - sin(x) using trigonometric identities
     *
     * @param x     The input value
     * @param delta The variance
     * @return The double output of the computation
     */
    private static double betterSinFunction(double x, double delta)
    {
        return 2 * Math.cos((2 * x + delta) / 2) * Math.sin(delta / 2);
    }

    /**
     * Computes sin(x) / x
     *
     * @param x the input value
     * @return The double output of the function
     */
    private static double sinc(double x)
    {
        return Math.sin(x) / x;
    }
}
