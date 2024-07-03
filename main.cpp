#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core.hpp>   
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <random>

using namespace std;
using namespace cv;

void fftshift(Mat& magnitude)
{
	// Rearrange the quadrants of the Fourier image
	int cx = magnitude.cols / 2;
	int cy = magnitude.rows / 2;

	Mat q0(magnitude, Rect(0, 0, cx, cy));
	Mat q1(magnitude, Rect(cx, 0, cx, cy));
	Mat q2(magnitude, Rect(0, cy, cx, cy));
	Mat q3(magnitude, Rect(cx, cy, cx, cy));

	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

Mat computePowerSpectrum(const Mat& signal)
{
	// Compute autocorrelation function
	Mat autocorrelation;
	matchTemplate(signal, signal, autocorrelation, TM_CCORR_NORMED);

	// Fourier transform of autocorrelation function
	Mat powerSpectrum;
	dft(autocorrelation, powerSpectrum, DFT_COMPLEX_OUTPUT);

	// Split into real and imaginary parts
	Mat planes[2];
	split(powerSpectrum, planes);

	// Calculate magnitude (power spectrum)
	Mat mag;
	magnitude(planes[0], planes[1], mag);

	// Shift the zero frequency component to the center
	fftshift(mag);

	return mag;
}

void calculateFFT(const Mat& inputImage, Mat& outputFFT) {
	Mat padded; // Expand image to optimal size
	int m = getOptimalDFTSize(inputImage.rows);
	int n = getOptimalDFTSize(inputImage.cols);
	copyMakeBorder(inputImage, padded, 0, m - inputImage.rows, 0, n - inputImage.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexImage;
	merge(planes, 2, complexImage);

	dft(complexImage, complexImage);

	// Shift the quadrants to center
	fftshift(complexImage);

	// Compute magnitude spectrum
	split(complexImage, planes);
	magnitude(planes[0], planes[1], outputFFT);
}

Mat computeNoisePowerSpectrum(const Mat& noisyImage)
{
	// Convert the input image to double precision
	Mat noisyImageDouble;
	noisyImage.convertTo(noisyImageDouble, CV_64F);

	// Assume noise is in a specific region of the image (you may need to adjust this)
	int noiseWidth = std::min(50, noisyImage.cols - 100);
	int noiseHeight = std::min(50, noisyImage.rows - 100);

	// Ensure that the region is even in size
	noiseWidth -= (noiseWidth % 2 == 1);
	noiseHeight -= (noiseHeight % 2 == 1);

	Rect noiseRegion(100, 100, noiseWidth, noiseHeight);
	Mat noiseRegionImage = noisyImageDouble(noiseRegion).clone();

	// Compute the FFT of the noise region
	Mat noiseRegionFFT;
	calculateFFT(noiseRegionImage, noiseRegionFFT);

	// Calculate the PSD by taking the squared magnitude of the FFT
	Mat psd;
	pow(noiseRegionFFT, 2, psd);

	// Shift the zero frequency component to the center
	fftshift(psd);

	return psd;
}


Mat generateGaussianPSF(Size size, double sigma)
{
	// Calculate the center of the PSF
	Point center(size.width / 2, size.height / 2);

	// Create an empty matrix for the PSF
	Mat psf(size, CV_64F);

	// Generate the Gaussian PSF
	for (int i = 0; i < size.width; ++i)
	{
		for (int j = 0; j < size.height; ++j)
		{
			double x = i - center.x;
			double y = j - center.y;
			psf.at<double>(j, i) = exp(-(x * x + y * y) / (2.0 * sigma * sigma));
		}
	}

	// Normalize the PSF
	normalize(psf, psf, 1.0, NORM_L1);

	return psf;
}

Mat LowLightEnhancment(Mat bgr_image, double val)
{
	Mat lab_image;
	cvtColor(bgr_image, lab_image, COLOR_BGR2Lab);
	Mat grayscale_image;
	cvtColor(bgr_image, grayscale_image, COLOR_BGR2GRAY);

	vector<Mat> lab_planes(3);
	split(lab_image, lab_planes);

	Ptr<CLAHE> clahe = createCLAHE();
	clahe->setClipLimit(val);
	Mat dst;
	clahe->apply(lab_planes[0], dst);

	dst.copyTo(lab_planes[0]);
	merge(lab_planes, lab_image);

	Mat image_clahe;
	cvtColor(lab_image, image_clahe, COLOR_Lab2BGR);

	return image_clahe;
}

Mat UnsharpMasking(Mat bgr_image, double alpha, int gamma)
{
	Mat blurred_image;
	GaussianBlur(bgr_image, blurred_image, Size(3, 3), 2.0);
	Mat unsharpMask = bgr_image - blurred_image;
	Mat sharpened;
	addWeighted(bgr_image, 1 + alpha, unsharpMask, -alpha, gamma, sharpened);
	return sharpened;
}

Mat BilateralFiltering(Mat bgr_image, int d, double sigma_color, double sigma_space)
{
	Mat filteredImage;
	bilateralFilter(bgr_image, filteredImage, d, sigma_color, sigma_space);
	return filteredImage;
}

Mat SaturationBoost(Mat image, double factor)
{
	Mat hsvImage;
	cvtColor(image, hsvImage, COLOR_BGR2HSV);
	//float saturationFactor = 3;
	hsvImage.forEach<Vec3b>([&factor](Vec3b& pixel, const int* position) -> void {pixel[1] = saturate_cast<uchar>(pixel[1] * factor); });

	Mat saturatedImage;
	cvtColor(hsvImage, saturatedImage, COLOR_HSV2BGR);
	return saturatedImage;
}

Mat HistogramEqalization(Mat image)
{
	Mat grayImage;
	cvtColor(image, grayImage, COLOR_BGR2GRAY);

	Mat equalizedImage;
	equalizeHist(grayImage, equalizedImage);
	return equalizedImage;
}

Mat CEH(Mat image)
{
	vector<Mat> channels;
	split(image, channels);
	for (int i = 0; i < channels.size(); ++i) {
		equalizeHist(channels[i], channels[i]);
	}
	Mat equalizedImage;
	merge(channels, equalizedImage);
	return equalizedImage;
}

string categorizeImage(const string& imagePath) {
	// Read the image
	Mat image = imread(imagePath);

	if (image.empty()) {
		cerr << "Error: Unable to load image " << imagePath << endl;
		return "";
	}

	// Convert the image to grayscale
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);

	// Calculate the Laplacian variance to determine sharpness
	Mat laplacian;
	Laplacian(gray, laplacian, CV_64F);
	Scalar mean, stddev;
	meanStdDev(laplacian, mean, stddev);
	double laplacianVar = stddev.val[0] * stddev.val[0];

	// Calculate the image brightness
	Scalar brightnessMean;
	Scalar brightnessStdDev;
	meanStdDev(gray, brightnessMean, brightnessStdDev);

	// Define thresholds for blur, lowlight, and sharpness
	double blurThreshold = 100.0;
	double lowlightThreshold = 100.0;
	double sharpnessThreshold = 100.0;

	// Categorize the image based on the thresholds
	if (brightnessMean[0] < lowlightThreshold) {
		return "Lowlight";
	}
	else if (laplacianVar < blurThreshold) {
		return "Blur";
	}
	else {
		return "Sharpening";
	}
}

double calculateBrightness(const Mat& image) {
	// Convert the image to grayscale
	Mat grayscale;
	cvtColor(image, grayscale, COLOR_BGR2GRAY);

	// Compute the average pixel intensity (brightness)
	Scalar avgIntensity = mean(grayscale);
	return avgIntensity[0]; // Return the average intensity value
}

double calculateLaplacianVariance(const Mat& image) {
	// Convert the image to grayscale
	Mat grayscale;
	cvtColor(image, grayscale, COLOR_BGR2GRAY);

	// Compute the Laplacian of the grayscale image
	Mat laplacian;
	Laplacian(grayscale, laplacian, CV_64F);

	// Compute the variance of the Laplacian
	Scalar mean, stddev;
	meanStdDev(laplacian, mean, stddev);
	double variance = stddev.val[0] * stddev.val[0];

	return variance;
}

Mat enhance(String Loc,String Name, String Ext) {
	String loc = Loc;
	String name = Name;
	String ext = Ext;

	//The string dir is the location of the input image.
	String dir = loc + name + ext;

	Mat bgr_image = imread(dir);

	//Categorising what sort of enhancement the image needs.
	string category = categorizeImage(dir);

	//Finding the score for brightness and bluriness. 
	double bright_val = calculateBrightness(bgr_image);
	double blur_val = calculateLaplacianVariance(bgr_image);
	
	double val = 0.0;
	double alpha_gaussian = 0.0;
	int gamma = 0;
	double sat_val = 1.1;

	Mat enhanced_image;

	//Starting the enhancements by first just adjusting the values based on the scores.
	if (bright_val < 80 && bright_val>30)
		val = 2.0;
	else if (bright_val < 30)
		val = 3.0;
	else
		val = 1.0;

	if (blur_val < 100)
	{
		alpha_gaussian = 1.0;
		gamma = -90;
	}
	else if(blur_val < 200)
	{
		alpha_gaussian = 0.5;
		gamma = -50;
	}
	else
	{
		alpha_gaussian = 0;
		gamma = 0;
		sat_val = 1.5;
	}

	//Applying the enhancments.
	enhanced_image = LowLightEnhancment(bgr_image, val);
	enhanced_image = UnsharpMasking(bgr_image, alpha_gaussian, gamma);
	enhanced_image = SaturationBoost(enhanced_image, sat_val);

	/*
	double scale_percent = .5; 
	int new_width = static_cast<int>(bgr_image.cols * scale_percent);
	int new_height = static_cast<int>(bgr_image.rows * scale_percent);

	Mat resized_og;
	resize(bgr_image, resized_og, Size(new_width, new_height));

	Mat resized_en;
	resize(enhanced_image, resized_en, Size(new_width, new_height));*/


	//Displaying all the values used and the classification.
	cout << category << endl;
	cout << "Brightness: " << bright_val << endl;
	cout << "Blurriness: " << blur_val << endl;
	cout << "Clip Limit: " << val << endl;
	cout << "Gamma: " << gamma << endl;
	cout << "Alpha: " << alpha_gaussian << endl;
	cout << "Sat Val: " << sat_val << endl;

	//Returning the enhanced image.
    return enhanced_image;
}

int main() 
{

	//The string loc is the location of the parent folder of the image.
	String loc = "";
	//The string name is the name of the image file.
	String name = "";
	//The string ext is the extension of the file ,.png,.jpg,.jpeg etc.
	String ext = "";

	Mat pic = imread(loc + name + ext);


	//Mat enhanced_image = enhance(loc, name, ext);
	Mat enhanced_image = BilateralFiltering(pic, 9, 75.0, 75.0);
	imshow("Original Image", pic);
	imshow("enhanced image", enhanced_image);

	Mat satstuff;
	satstuff = SaturationBoost(enhanced_image, 2.0);
	imshow("sat", satstuff);
	
	//The dir string is the entire file path of the enhanced image.
	//This can be anything set by the user and the program will automatically create the image in your system.
	String dir = loc + name + "-en" + ext;
	String dir2 = loc + name + "-en2" + ext;
	imwrite(dir,enhanced_image);
	imwrite(dir2, satstuff);

	waitKey();

}
