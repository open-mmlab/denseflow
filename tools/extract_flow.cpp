#include "dense_flow.h"
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
		};

	CommandLineParser cmd(argc, argv, keys);
	string vidFile = cmd.get<string>("vidFile");
	string xFlowFile = cmd.get<string>("xFlowFile");
	string yFlowFile = cmd.get<string>("yFlowFile");
	string imgFile = cmd.get<string>("imgFile");
	int bound = cmd.get<int>("bound");

	vector<vector<uchar> > out_vec_x, out_vec_y, out_vec_img;

	calcDenseFlow(vidFile, bound, 0, 1,
					 out_vec_x, out_vec_y, out_vec_img);

	return 0;
}
