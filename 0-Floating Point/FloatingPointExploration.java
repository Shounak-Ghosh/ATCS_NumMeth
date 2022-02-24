/**
 * @author Shounak Ghosh
 * @version 1/23/22
 * <p>
 * This code runs to determine the limits of the Java language and compiler.
 * <p>
 * This the following transfinite arithmetic tests are run:
 * nonzero / zero
 * zero / zero
 * ±Infinity / zero
 * zero / ±Infinity
 * zero × ±Infinity
 * ±Infinity × ±Infinity
 * ±Infinity / ±Infinity
 * <p>
 * We then determine the values of ±MIN and ±MAX using the algorithms below:
 * 2 × (MIN / 2) ≠ 0
 * and
 * (MAX × 2) / 2  ≠ Infinity
 * and EPS
 * 1 + (EPS/2) = 1 for EPS ≠ 0
 */
public class FloatingPointExploration
{
    private static final double ZERO = 0.0;
    private static final double NONZERO = 1.0;

    /**
     * Driver method
     *
     * @param args stores command line arguments
     */
    public static void main(String[] args)
    {
        System.out.println("nonzero / zero: " + NONZERO / ZERO);
        System.out.println("zero / zero: " + ZERO / ZERO);

        int z = 0;

        //System.out.println(z/z);

        double posInfinity = NONZERO / ZERO;
        double negInfinity = -NONZERO / ZERO;

        System.out.println("Infinity / zero: " + (posInfinity / ZERO + "\n-Infinity / zero: " + negInfinity / ZERO));
        System.out.println("zero / Infinity: " + ZERO / posInfinity + "\nzero / -Infinity: " + ZERO / negInfinity);
        System.out.println("zero * Infinity: " + ZERO * posInfinity + "\nzero * -Infinity: " + ZERO * negInfinity);

        System.out.println("-Infinity * -Infinity: " + negInfinity * negInfinity + "\n-Infinity "
                + "*" + " Infinity: " + negInfinity * posInfinity + "\nInfinity * -Infinity: "
                + posInfinity * negInfinity + "\nInfinity * Infinity: "
                + posInfinity * posInfinity);

        System.out.println("-Infinity / -Infinity: " + negInfinity / negInfinity + "\n-Infinity "
                + "/" + " Infinity: " + negInfinity / posInfinity + "\nInfinity / -Infinity: "
                + posInfinity / negInfinity + "\nInfinity / Infinity: "
                + posInfinity / posInfinity + "\n");

        System.out.println(negInfinity / negInfinity == ZERO * posInfinity);

        double min = 1.0;
        while (2 * (min / 2) != 0)
        {
            min /= 2;
        }
        System.out.println("±MIN: " + min);

        double max = 1.0;
        while ((max * 2) / 2 != posInfinity)
        {
            max *= 2;
        }
        System.out.println("±MAX: " + max);

        double eps = 1.0;
        while (1 + (eps / 2) != 1)
        {
            eps /= 2;
        }
        System.out.println("±EPS: " + eps);
    }
}
