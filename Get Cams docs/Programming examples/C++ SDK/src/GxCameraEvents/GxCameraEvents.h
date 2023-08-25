// GxCameraEvents.h : main header file for the GxCameraEvents application
//

#pragma once

#ifndef __AFXWIN_H__
	#error include 'stdafx.h' before including this file for PCH
#endif

#include "resource.h"		// main symbols


// CGxCameraEventsApp:
// See GxCameraEvents.cpp for the implementation of this class
//

class CGxCameraEventsApp : public CWinApp
{
public:
	CGxCameraEventsApp();

// Overrides
	public:
	virtual BOOL InitInstance();

// Implementation

	DECLARE_MESSAGE_MAP()
};

extern CGxCameraEventsApp theApp;