/*	=========================================================================
	Author: Leonardo Citraro
	Company: 
	Filename: .hpp
	Last modifed:   06.12.2016 by Leonardo Citraro
	Description:    Attempt to track people
	=========================================================================
	=========================================================================
*/
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/background_segm.hpp"
#include <iostream>
#include <algorithm>

using namespace cv;
using namespace std;


Mat element1 = getStructuringElement( MORPH_ELLIPSE, Size(5,5));
Mat element2 = getStructuringElement( MORPH_ELLIPSE, Size(10,10));	

RNG rng(12345);													

class Person{
	
	private:
		const unsigned int 	_N = 5; // remember the last N settings
		vector<Point> 		_o;		// box origin (upper left corner)
		vector<Point> 		_e;		// box end (bottom right corner)
		Point 				_mean_o;
		Point 				_mean_e;
		const unsigned int	_n_hist = 256;
		vector<Mat>			_hists;
		Scalar 				_color;
		int 				_last_update = 0;
		
	public:
		Person(){
			set_random_color();
		}
		Person(vector<Mat>& hists) : _hists(hists){
			 set_random_color(); 
		}
		Person(Point o, Point e, vector<Mat>& hists) : _hists(hists){
			push_o(o);
			push_e(e);
			set_random_color(); 
		}
		~Person(){}	
		
		Person& operator=(const Person& p){
			this->_o = p._o;
			this->_e = p._e;
			this->_mean_o = p._mean_o;
			this->_mean_e = p._mean_e;
			this->_hists = p._hists;
			this->_color = p._color;
		}
		
		void push_o(Point p){
			_o.push_back(p);
			if(_o.size() > _N)
				_o.erase( _o.begin() );
			_mean_o = mean_point(_o);
		}
		void push_e(Point p){
			_e.push_back(p);
			if(_e.size() > _N)
				_e.erase( _e.begin() );
			_mean_e = mean_point(_e);
		}
		
		Point get_mean_o(){return _mean_o;}
		Point get_mean_e(){return _mean_e;}
		
		double distance(Person& p){
			double dist = 0;
			for(size_t h=0; h<_hists.size(); ++h)
				dist += (double)mean(abs(p.get_hists()[h]-_hists[h]))[0];
			//~ return dist*cv::norm(_mean_o-p._mean_o);
			return dist;
		}
		
		void set_hists(vector<Mat> hists){ _hists = hists; }		
		vector<Mat> get_hists(){ return _hists; }
		
		void draw_rectangle(Mat& img){
			rectangle(img, _mean_o, _mean_e, _color, 2, 8, 0);
		}
		
		void set_random_color(){
			_color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
		}
		
		void update(Person p, int n_frame){
			_last_update = n_frame;
			//this->set_hists(p.get_hists());
			this->push_o(p.get_mean_o());
			this->push_e(p.get_mean_e());
		}
		
		bool is_perons_in_box(Mat& binary){
			Mat sub_binary = binary(cv::Range(_mean_o.y, _mean_e.y), cv::Range(_mean_o.x, _mean_e.x));
			double sum = cv::sum(sub_binary)[0];
			int area = sub_binary.rows * sub_binary.cols;
			return sum/area > 0.4 ? true : false;
		}
		
		static Point mean_point(vector<Point>& ps){
			unsigned int x = 0;
			unsigned int y = 0;
			for(size_t i=0; i<ps.size(); ++i){
					x += ps[i].x;
					y += ps[i].y;				
			}
			return Point((int)(x/ps.size()), (int)(y/ps.size()));
		}
};

class Scene{
	private:
		Mat current_frame;
		Mat binary;
		vector<Person> People;
		int 	history			= 1000;
		double 	dist2Threshold	= 100.0;
		bool 	detectShadows	= false;
		Ptr<BackgroundSubtractor> pKNN;
		Mat labels, stats, centroids;
		int connectivity = 8;	
		int components = 0;	
													
		
	public:
		Scene(){
			pKNN = createBackgroundSubtractorKNN(history, dist2Threshold, detectShadows);
		}
		~Scene(){}
		
		void update(Mat& frame, int n_frame, int min_connected_pixels, int max_connected_pixels, double min_distance){
			
			current_frame = frame;
			pKNN->apply(current_frame, binary);
				
			morphologyEx( binary, binary, MORPH_OPEN, element1 );
			//~ erode( binary, binary, element1 );
			dilate( binary, binary, element2 );
			//~ blur( binary, binary, Size(20,20) );
			//~ threshold(binary, binary, 100.0, 255., CV_THRESH_BINARY);

			components =  connectedComponentsWithStats(binary, labels, stats, centroids, connectivity, CV_32S);

			vector<int> updated;
			for( size_t i=1; i<components; ++i){
				if(stats.at<int>(i,4) > min_connected_pixels && stats.at<int>(i,4) < max_connected_pixels){
					
					// get position
					Point o = Point(stats.at<int>(i,0), stats.at<int>(i,1)); // origin
					Point e = o + Point(stats.at<int>(i,2), stats.at<int>(i,3)); // end
					
					// get histogram of the person ith
					vector<Mat> bgr_planes;
					vector<Mat> hists(3);
					split( current_frame, bgr_planes );
					Mat mask = (labels==i);
					const int histSize = 256;
					float range[] = { 0, 256 } ;
					const float* histRange = { range };
					bool uniform = true; 
					bool accumulate = false;
					calcHist(&bgr_planes[0], 1, 0, mask, hists[0], 1, &histSize, &histRange, uniform, accumulate);
					calcHist(&bgr_planes[1], 1, 0, mask, hists[1], 1, &histSize, &histRange, uniform, accumulate);
					calcHist(&bgr_planes[2], 1, 0, mask, hists[2], 1, &histSize, &histRange, uniform, accumulate);
					
					//~ normalize(hists[0], hists[0], 0, 1, NORM_MINMAX, -1, Mat() );
					//~ normalize(hists[1], hists[1], 0, 1, NORM_MINMAX, -1, Mat() );
					//~ normalize(hists[2], hists[2], 0, 1, NORM_MINMAX, -1, Mat() );
					
					Person temp = Person(o,e,hists);	
					
					vector<double> distances;
					for(auto p : People){
						distances.push_back( p.distance(temp) );
					}
					
					cout << "Peoples: " << People.size() << endl;
					
					auto it_min = min_element( distances.begin(), distances.end() );
					int pos = std::distance(distances.begin(), it_min);	
					
					cout << "Pos min: " << pos << endl;
					if(People.size() != 0)cout << "Dist: " << distances[pos] << endl;
						
					if( People.size() == 0 || distances[pos] > min_distance ) People.push_back(temp);
					else {
						People[pos].update(temp, n_frame);
						updated.push_back(pos);
					}
				
				}
				
			}
			
			//~ vector<int> to_erase;
			//~ for(size_t p=0; p<People.size(); p++){
				//~ bool is_updated = false;
				//~ for(size_t i=0; i<updated.size(); i++){
					//~ if(p==updated[i]) is_updated=true;
				//~ }
				//~ if(is_updated == false) to_erase.push_back(p);
			//~ }
			//~ for(size_t p=0; p<to_erase.size(); p++){
				//~ People.erase(People.begin()+to_erase[p]);
			//~ }
			
			vector<int> to_erase;
			for(size_t p=0; p<People.size(); p++){
				if(People[p].is_perons_in_box(binary) == false) to_erase.push_back(p);
			}
			for(size_t p=0; p<to_erase.size(); p++){
				People.erase(People.begin()+to_erase[p]);
			}
			
										
		}
		
		void display(){
			imshow("Original", current_frame);
			imshow("Binary", binary);
			Mat boxes = current_frame.clone();
			for(size_t i=0; i<People.size(); ++i)
				People[i].draw_rectangle(boxes);
			imshow("Boxes", boxes);			
		}		
};

int main(int argc, char* argv[])
{
    //VideoCapture cap("./tennis_sample.mp4");
	VideoCapture cap(argv[1]);

    if ( !cap.isOpened() ){
         cout << "Cannot open the video file" << endl;
         return -1;
    }

    //cap.set(CV_CAP_PROP_POS_MSEC, 300); //start the video at 300ms
	int width = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_WIDTH));
	int height = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	int n_frame = 0;
	
	//create a windows
	namedWindow("Original",WINDOW_NORMAL); 
	namedWindow("Binary",WINDOW_AUTOSIZE);
	namedWindow("Boxes",WINDOW_AUTOSIZE);

	//~ int Threshold = 20;
	//~ const int max_Threshold = 255;
	//~ createTrackbar( "Threshold:", "Subtraction", &Threshold, max_Threshold );

	int min_connected_pixels = 400;
	const int Max_min_connected_pixels = 4000;
	createTrackbar( "Min connected pixels:", "Boxes", &min_connected_pixels, Max_min_connected_pixels );
	
	int max_connected_pixels = 10000;
	const int Max_max_connected_pixels = 50000;
	createTrackbar( "Max connected pixels:", "Boxes", &max_connected_pixels, Max_max_connected_pixels );
	
	int min_distance = 30;
	const int Max_min_distance = 1000;
	createTrackbar( "Min distance:", "Boxes", &min_distance, Max_min_distance );
	
	Mat frame_new;
	
	Scene scene = Scene();
	
    while(1){

		n_frame = cap.get(CV_CAP_PROP_POS_FRAMES);
		//cout << n_frame  << endl;		

		if (!cap.read(frame_new)){
			cout << "Cannot read the frame from video file" << endl;
			break;
		}
		
		scene.update(frame_new, n_frame, min_connected_pixels, max_connected_pixels, min_distance);
		scene.display();

		//wait for 'esc' key press for 30 ms
        if(waitKey(30) == 27){
			cout << "esc key is pressed by user" << endl; 
			break; 
       }

    }

    return 0;

}
