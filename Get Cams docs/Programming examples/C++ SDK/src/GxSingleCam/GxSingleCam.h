// GxSingleCam.h : PROJECT_NAME Heard file of the application
//

#pragma once

#ifndef __AFXWIN_H__
	#error include 'stdafx.h' before including this file for PCH
#endif

#include "resource.h"		// main symbols


// CGxSingleCamApp:
// See GxSingleCam.cpp for the implementation of this class.
//

class CGxSingleCamApp : public CWinApp
{
public:
	CGxSingleCamApp();

// Overrides
	public:
	virtual BOOL InitInstance();

// Implementation

	DECLARE_MESSAGE_MAP()
};

extern CGxSingleCamApp theApp;