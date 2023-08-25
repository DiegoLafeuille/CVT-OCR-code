//----------------------------------------------------------------------------------
/**
\file    Main.cpp 
\brief   Gige recovery function
*/
//----------------------------------------------------------------------------------
#include "stdafx.h"
#include "GxGigeRecovery.h"
int _tmain(int argc, _TCHAR* argv[])
{
	// Call the CGigeRecovery class for demonstrating how to recovery from the offLine status 
	CGigeRecovery objCtrDevice;
	objCtrDevice.GigeRecovery();

	// Press any key to exit the console application
	std::cout<<"\n<press any key to exit>"<<std::endl;
	while (!_kbhit());

	return 0;
}