#include "dense_flow.h"
#include "utils.h"
INITIALIZE_EASYLOGGINGPP

int main(int argc, char** argv)
{
	// IO operation

	const char* keys =
		{
			"{ f  | vidFile      | ex2.avi | filename of video }"
			"{ x  | xFlowFile    | flow_x | filename of flow x component }"
			"{ y  | yFlowFile    | flow_y | filename of flow x component }"
			"{ i  | imgFile      | flow_i | filename of flow image}"
			"{ b  | bound | 15 | specify the maximum of optical flow}"
			"{ o  | out | zip | output style}"
		};

	CommandLineParser cmd(argc, argv, keys);
	string vidFile = cmd.get<string>("vidFile");
	string xFlowFile = cmd.get<string>("xFlowFile");
	string yFlowFile = cmd.get<string>("yFlowFile");
	string imgFile = cmd.get<string>("imgFile");
	string output_style = cmd.get<string>("out");
	int bound = cmd.get<int>("bound");

//	LOG(INFO)<<"Starting extraction";
	vector<vector<uchar> > out_vec_x, out_vec_y, out_vec_img;

	calcDenseFlow(vidFile, bound, 0, 1,
					 out_vec_x, out_vec_y, out_vec_img);

	if (output_style == "dir") {
		writeImages(out_vec_x, xFlowFile);
		writeImages(out_vec_y, yFlowFile);
		writeImages(out_vec_img, imgFile);
	}else{
//		LOG(INFO)<<"Writing results to Zip archives";
		writeZipFile(out_vec_x, "x_%05d.jpg", xFlowFile+".zip");
		writeZipFile(out_vec_y, "y_%05d.jpg", yFlowFile+".zip");
		writeZipFile(out_vec_img, "img_%05d.jpg", imgFile+".zip");
	}
	return 0;
}
