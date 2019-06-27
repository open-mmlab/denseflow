#include "dense_flow.h"
#include "utils.h"
INITIALIZE_EASYLOGGINGPP

int main(int argc, char** argv)
{
	// IO operation

	const char* keys =
		{
			"{ f vidFile   | ex2.avi | filename of video            }"
			"{ x xFlowFile | flow_x  | filename of flow x component }"
			"{ y yFlowFile | flow_y  | filename of flow x component }"
			"{ i imgFile   | flow_i  | filename of flow image       }"
			"{ b bound     | 15      | specify the maximum of optical flow}"
			"{ t type      | 0       | specify the optical flow algorithm }"
			"{ o out       | zip     | output style                 }"
		};

	CommandLineParser cmd(argc, argv, keys);
	std::string vidFile = cmd.get<std::string>("vidFile");
	std::string xFlowFile = cmd.get<std::string>("xFlowFile");
	std::string yFlowFile = cmd.get<std::string>("yFlowFile");
	std::string imgFile = cmd.get<std::string>("imgFile");
	std::string output_style = cmd.get<std::string>("out");
	int bound = cmd.get<int>("bound");
    int type  = cmd.get<int>("type");

//	LOG(INFO)<<"Starting extraction";
	std::vector<std::vector<uchar> > out_vec_x, out_vec_y, out_vec_img;

	calcDenseFlow(vidFile, bound, type, 1,
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
