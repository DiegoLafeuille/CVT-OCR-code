// GxMultiCam.h : main header file for the GxMultiCam application
//
#pragma once

#ifndef __AFXWIN_H__
	#error "include 'stdafx.h' before including this file for PCH"
#endif

#include "resource.h"       // main symbols


// CGxMultiCamApp:
// See GxMultiCam.cpp for the implementation of this class
//

class CGxMuitCamApp : public CWinApp
{
public:
	CGxMuitCamApp();


// Overrides
public:
	virtual BOOL InitInstance();

// Implementation
	afx_msg void OnAppAbout();
	DECLARE_MESSAGE_MAP()
};

extern CGxMuitCamApp theApp;