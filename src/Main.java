/**
 * Vecsei Gábor
 *
 * Blog:        https://gaborvecsei.wordpress.com/
 * Email:       vecseigabor.x@gmail.com
 * LinkedIn:    https://hu.linkedin.com/in/vecsei-gábor-004b8611a
 *
 * 2016.08.29.
 */

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class Main {

    public static void main(String[] args) {
        //For OpenCV (this is compulsory)
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Main m = new Main();
        //read our original image
        Mat image = Imgcodecs.imread("test_image.jpg");

        //Straight it out! :)
        Mat straightImage = m.straightenImage(image);
        Imgcodecs.imwrite("straightImage.jpg", straightImage);
    }

    //This is the pre-processing part where we create a binary image from our original
    //And after the morphology we can detect the test parts more easily
    private Mat preProcessForAngleDetection(Mat image) {
        Mat binary = new Mat();
        //Create binary image
        Imgproc.threshold(image, binary, 127, 255, Imgproc.THRESH_BINARY_INV);
        //"Connect" the letters and words
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(20, 1));
        Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_CLOSE, kernel);
        //Convert the image to gray from RGB
        Imgproc.cvtColor(binary, binary, Imgproc.COLOR_BGR2GRAY);
        Imgcodecs.imwrite("processedImage.jpg", binary);
        return binary;
    }

    //With this we can detect the rotation angle
    //After this function returns we will know the necessary angle
    private double detectRotationAngle(Mat binaryImage) {
        //Store line detections here
        Mat lines = new Mat();
        //Detect lines
        Imgproc.HoughLinesP(binaryImage, lines, 1, Math.PI / 180, 100);

        double angle = 0;

        //This is only for debugging and to visualise the process of the straightening
        Mat debugImage = binaryImage.clone();
        Imgproc.cvtColor(debugImage, debugImage, Imgproc.COLOR_GRAY2BGR);

        //Calculate the start and end point and the angle
        for (int x = 0; x < lines.cols(); x++) {
            double[] vec = lines.get(0, x);
            double x1 = vec[0];
            double y1 = vec[1];
            double x2 = vec[2];
            double y2 = vec[3];

            Point start = new Point(x1, y1);
            Point end = new Point(x2, y2);

            //Draw line on the "debug" image for visualization
            Imgproc.line(debugImage, start, end, new Scalar(255, 255, 0), 5);

            //Calculate the angle we need
            angle = calculateAngleFromPoints(start, end);
        }

        Imgcodecs.imwrite("detectedLines.jpg", debugImage);

        return angle;
    }

    //From an end point and from a start point we can calculate the angle
    private double calculateAngleFromPoints(Point start, Point end) {
        double deltaX = end.x - start.x;
        double deltaY = end.y - start.y;
        return Math.atan2(deltaY, deltaX) * (180 / Math.PI);
    }

    //Rotation is done here
    private Mat rotateImage(Mat image, double angle) {
        //Calculate image center
        Point imgCenter = new Point(image.cols() / 2, image.rows() / 2);
        //Get the rotation matrix
        Mat rotMtx = Imgproc.getRotationMatrix2D(imgCenter, angle, 1.0);
        //Calculate the bounding box for the new image after the rotation (without this it would be cropped)
        Rect bbox = new RotatedRect(imgCenter, image.size(), angle).boundingRect();

        //Rotate the image
        Mat rotatedImage = image.clone();
        Imgproc.warpAffine(image, rotatedImage, rotMtx, bbox.size());

        return rotatedImage;
    }

    //Sums the whole process and returns with the straight image
    private Mat straightenImage(Mat image) {
        Mat rotatedImage = image.clone();
        Mat processed = preProcessForAngleDetection(image);
        double rotationAngle = detectRotationAngle(processed);

        return rotateImage(rotatedImage, rotationAngle);
    }


}
