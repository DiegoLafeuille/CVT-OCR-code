// GxCameraEventsDlg.cpp : implementation file
//

#include "stdafx.h"
#include "GxCameraEvents.h"
#include "GxCameraEventsDlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CAboutDlg dialog used for App About

class CAboutDlg : public CDialog
{
public:
	CAboutDlg();

// Dialog Data
	//{{AFX_DATA(CAboutDlg)
	enum { IDD = IDD_ABOUTBOX };
	CString	m_strProductVersion;
	CString	m_strLegalCopyRight;
	//}}AFX_DATA

	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CAboutDlg)
	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
	//}}AFX_VIRTUAL

// Implementation
protected:
	//{{AFX_MSG(CAboutDlg)
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialog(CAboutDlg::IDD)
{
	//{{AFX_DATA_INIT(CAboutDlg)
	m_strProductVersion = _T("");
	m_strLegalCopyRight = _T("");
	//}}AFX_DATA_INIT
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CAboutDlg)
	DDX_Text(pDX, IDC_STATIC_PRODUCTVERSION, m_strProductVersion);
	DDX_Text(pDX, IDC_STATIC_LEGALCOPYRIGHT, m_strLegalCopyRight);
	//}}AFX_DATA_MAP
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialog)
	//{{AFX_MSG_MAP(CAboutDlg)
		// No message handlers
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CGxCameraEventsDlg dialog

CGxCameraEventsDlg::CGxCameraEventsDlg(CWnd* pParent /*=NULL*/)
	: CDialog(CGxCameraEventsDlg::IDD, pParent)
{
	//{{AFX_DATA_INIT(CGxCameraEventsDlg)
		// NOTE: the ClassWizard will add member initialization here
	//}}AFX_DATA_INIT
	// Note that LoadIcon does not require a subsequent DestroyIcon in Win32
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
	m_hDevice                     = NULL;            // The handle for device 
	m_pLogWnd                     = NULL;            // The pointer for recording the time of the event
	m_bIsOpen                     = false;           // The flag indicates whether the first device has been opened
	m_bIsSnap                     = false;           // The flag indicates whether AcquisitionStart command has been send 
	m_bImplementEventSelector     = false;           // The flag indicates whether the device supports EventSourceSelector 
	m_bSupportExposureEndEvent    = false;           // The flag indicates whether the device supports ExposureEndEvent
	m_bIsTrigValid                = true;            // The flag indicates Whether the trigger is valid or not
}

void CGxCameraEventsDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CGxCameraEventsDlg)
		// NOTE: the ClassWizard will add DDX and DDV calls here
	//}}AFX_DATA_MAP
}

BEGIN_MESSAGE_MAP(CGxCameraEventsDlg, CDialog)
	//{{AFX_MSG_MAP(CGxCameraEventsDlg)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_WM_CLOSE()
	ON_BN_CLICKED(IDC_OPEN_DEVICE, OnOpenDevice)
	ON_BN_CLICKED(IDC_START_SNAP, OnStartSnap)
	ON_BN_CLICKED(IDC_STOP_SNAP, OnStopSnap)
	ON_BN_CLICKED(IDC_CLOSE_DEVICE, OnCloseDevice)
	ON_BN_CLICKED(IDC_SEND_SOFT_COMMAND, OnSendSoftCommand)
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CGxCameraEventsDlg message handlers

BOOL CGxCameraEventsDlg::OnInitDialog()
{
	CDialog::OnInitDialog();

	// Add "About..." menu item to system menu.

	// IDM_ABOUTBOX must be in the system command range.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		CString strAboutMenu;
		strAboutMenu.LoadString(IDS_ABOUTBOX);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon
	
	// TODO: Add extra initialization here
	// Before using any GxIAPI methods, the GxIAPI must be initialized. 
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	emStatus = GXInitLib();
	if (emStatus != GX_STATUS_SUCCESS)
	{
		ShowErrorString(emStatus);
		exit(0);
	}

	// The pointer for showing the time to the window
	m_pLogWnd = GetDlgItem(IDC_EDIT_TIME_LOG); 

	UpdateUI();	
	
	return TRUE;  // return TRUE  unless you set the focus to a control
}

void CGxCameraEventsDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		char chFileName[MAX_PATH] = {'\0'};
		GetModuleFileName(NULL, chFileName, MAX_PATH);
		CFileVersion fv(chFileName);
		CAboutDlg dlgAbout;		
		dlgAbout.m_strProductVersion = _T("Version: ") + fv.GetProductVersion();
		dlgAbout.m_strLegalCopyRight = fv.GetLegalCopyright();
		dlgAbout.DoModal();
	}
	else
	{
		CDialog::OnSysCommand(nID, lParam);
	}
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CGxCameraEventsDlg::OnPaint() 
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, (WPARAM) dc.GetSafeHdc(), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialog::OnPaint();
	}
}

// The system calls this to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CGxCameraEventsDlg::OnQueryDragIcon()
{
	return (HCURSOR) m_hIcon;
}

void CGxCameraEventsDlg::OnClose() 
{
	// TODO: Add your message handler code here and/or call default
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	// If the device is acquiring image then stop it.
	if (m_bIsSnap)
	{
		// Send AcquisitionStop command
		emStatus = GXSendCommand(m_hDevice, GX_COMMAND_ACQUISITION_STOP);
		
		// Unregister the CaptureCallback function 
		emStatus = GXUnregisterCaptureCallback(m_hDevice);

		// Unregister the ExposureEndEvent callback function
		emStatus = GXUnregisterFeatureCallback(m_hDevice, GX_INT_EVENT_EXPOSUREEND, m_hCB);			
 		m_bIsSnap = false;
	}
	
	if (m_bIsOpen)
	{
		emStatus = GXCloseDevice(m_hDevice);
		m_bIsOpen = false;
		m_hDevice = NULL;
	}		
	
	// Release GxIAPI
	emStatus = GXCloseLib();
	
	CDialog::OnClose();
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to enumerate all camera devices and initialize device.

\return void
*/
//----------------------------------------------------------------------------------
void CGxCameraEventsDlg::OnOpenDevice() 
{
	// TODO: Add your control notification handler code here
	GX_STATUS      emStatus     = GX_STATUS_SUCCESS;
	uint32_t       nDeviceNum   = 0;
	int64_t        nPixelFormat = 0;        
	GX_OPEN_PARAM  stOpenParam;

	// Enumerate all camera devices 
	emStatus = GXUpdateDeviceList(&nDeviceNum, 1000);
	GX_VERIFY(emStatus);
	
	// Check if found any device
	if (nDeviceNum <= 0)
	{
		MessageBox("No devices found!");
		return;
	}

	// If the device is opened then close it to ensure the camera could open again.
	if (m_hDevice != NULL)
	{
		emStatus = GXCloseDevice(m_hDevice);
		GX_VERIFY(emStatus);
		m_hDevice = NULL;
	}
	
	stOpenParam.accessMode = GX_ACCESS_EXCLUSIVE;
	stOpenParam.openMode   = GX_OPEN_INDEX;
	stOpenParam.pszContent = "1";
	//Open the first found device 
	emStatus = GXOpenDevice(&stOpenParam, &m_hDevice);
	GX_VERIFY(emStatus);
    
    // It is suggested that users set the camera's packet size according to the current network environment after opening the network camera.
    // To improve the acquisition performance of network cameras, the following code is used for reference.
    {
        bool	 bImplementPacketSize = false;
        uint32_t unPacketSize		  = 0;

        emStatus = GXIsImplemented(m_hDevice, GX_INT_GEV_PACKETSIZE, &bImplementPacketSize);
        GX_VERIFY(emStatus);

        if (bImplementPacketSize)
        {
            emStatus = GXGetOptimalPacketSize(m_hDevice,&unPacketSize);
            GX_VERIFY(emStatus);

            emStatus = GXSetInt(m_hDevice, GX_INT_GEV_PACKETSIZE, unPacketSize);
            GX_VERIFY(emStatus);
        }
    }

	emStatus = InitParam();
	GX_VERIFY(emStatus);

	if (!m_bSupportExposureEndEvent)
	{
		return;
	}

	m_bIsOpen = true;	
	
	UpdateUI();	
}

//----------------------------------------------------------------------------------
/*
\brief  This function demonstrates how to start acquisition

\return void
*/
//----------------------------------------------------------------------------------
void CGxCameraEventsDlg::OnStartSnap() 
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	// Register ExposureEndEvent callback function
	emStatus = GXRegisterFeatureCallback(m_hDevice, this, OnFeatureCallbackFun, GX_INT_EVENT_EXPOSUREEND, &m_hCB);
	GX_VERIFY(emStatus);
		
	// Register the CaptureCallback function
	emStatus = GXRegisterCaptureCallback(m_hDevice, this, OnFrameCallbackFun);
	GX_VERIFY(emStatus);

	//Set StreamBufferHandlingMode
	emStatus = GXSetEnum(m_hDevice, GX_DS_ENUM_STREAM_BUFFER_HANDLING_MODE, GX_DS_STREAM_BUFFER_HANDLING_MODE_OLDEST_FIRST);
	GX_VERIFY(emStatus);
	
	// Send AcquisitionStart command
	emStatus = GXSendCommand(m_hDevice, GX_COMMAND_ACQUISITION_START);
	GX_VERIFY(emStatus);
	m_bIsSnap = true;
	m_bIsTrigValid = true;

	UpdateUI();	
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to stop acquisition
\return void
*/
//----------------------------------------------------------------------------------
void CGxCameraEventsDlg::OnStopSnap() 
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	
	// Send AcquisitionStop command
	emStatus = GXSendCommand(m_hDevice, GX_COMMAND_ACQUISITION_STOP);
	GX_VERIFY(emStatus);
	
	// Unregister the CaptureCallback function
	emStatus = GXUnregisterCaptureCallback(m_hDevice);
	GX_VERIFY(emStatus);

	// Unregister the ExposureEndEvent callback function
 	emStatus = GXUnregisterFeatureCallback(m_hDevice, GX_INT_EVENT_EXPOSUREEND, m_hCB);
 	GX_VERIFY(emStatus);
	m_bIsSnap = false;
	
	UpdateUI();
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to close device.
\return void
*/
//----------------------------------------------------------------------------------
void CGxCameraEventsDlg::OnCloseDevice() 
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	// If the device is acquiring image then stop it.
	if (m_bIsSnap)
	{
		// Send AcquisitionStop command
		emStatus = GXSendCommand(m_hDevice, GX_COMMAND_ACQUISITION_STOP);
		if (emStatus != GX_STATUS_SUCCESS)
		{
			// TODO: Add your error handler code here  
		}
		
		// Unregister the CaptureCallback function
		emStatus = GXUnregisterCaptureCallback(m_hDevice);
		if (emStatus != GX_STATUS_SUCCESS)
		{
			// TODO: Add your error handler code here  
		}

		// Unregister the ExposureEndEvent callback function
		emStatus = GXUnregisterFeatureCallback(m_hDevice, GX_INT_EVENT_EXPOSUREEND, m_hCB);
		if (emStatus != GX_STATUS_SUCCESS)
		{
			// TODO: Add your error handler code here  
		}
		m_bIsSnap = false;
	}
	
	emStatus = GXCloseDevice(m_hDevice);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		// TODO: Add your error handler code here  
	}
	m_hDevice = NULL;
	m_bIsOpen = false;
	
	UpdateUI();	
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to send TriggerSoftware command

\return void
*/
//----------------------------------------------------------------------------------
void CGxCameraEventsDlg::OnSendSoftCommand() 
{
	// TODO: Add your control notification handler code here
	// If you click the trigger button again while the trigger callback is executing, the click will be blocked
	if (!m_bIsTrigValid)
	{
		return;
	}
	else
	{
		m_bIsTrigValid = false;
	}

	SYSTEMTIME system;
	CString    strTime = "";
	GX_STATUS  emStatus = GX_STATUS_SUCCESS;

	// Get the time of sending TriggerSoftware command 
	GetLocalTime(&system);
	strTime.Format("%02dh:%02dm:%02ds:%03dms ", system.wHour, system.wMinute, system.wSecond, system.wMilliseconds);
	PrintInfotoWnd(MsgComposer("\r\nThe time of sending TriggerSoftware command:" + strTime));

	// Start timing for receiving image
	m_objImageTime.Start();

	// Start timing for recording the process of ExposureEndEvent
	m_objShutterTime.Start();

	// Send TriggerSoftware command
	emStatus = GXSendCommand(m_hDevice,GX_COMMAND_TRIGGER_SOFTWARE);
	if(emStatus != GX_STATUS_SUCCESS)
	{
		ShowErrorString(emStatus); 
		m_bIsTrigValid = true;
		return;
	} 
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize device 

\return  emStatus GX_STATUS_SUCCESS: Success, Else: Fail
*/
//----------------------------------------------------------------------------------
GX_STATUS CGxCameraEventsDlg::InitParam()
{
	GX_STATUS emStatus    = GX_STATUS_SUCCESS;

	// Set the continuous frame acquisition mode
	emStatus = GXSetEnum(m_hDevice, GX_ENUM_ACQUISITION_MODE, GX_ACQ_MODE_CONTINUOUS);
	VERIFY_STATUS_RET(emStatus);

	// Set the TriggerMode on
	emStatus = GXSetEnum(m_hDevice,GX_ENUM_TRIGGER_MODE, GX_TRIGGER_MODE_ON);
	VERIFY_STATUS_RET(emStatus);

	// Set the TriggerSource to SoftTrigger
	emStatus = GXSetEnum(m_hDevice,GX_ENUM_TRIGGER_SOURCE, GX_TRIGGER_SOURCE_SOFTWARE);
	VERIFY_STATUS_RET(emStatus);

	// Check whether the device supports ExposureEndEvent
	emStatus = __IsSupportEvent(m_bSupportExposureEndEvent);
	if (!m_bSupportExposureEndEvent)
	{
        MessageBox("This sample is invalid because of the camera doesn't support ExposureEndEvent !"); 
		return emStatus;
	}
	
	// Select the ExposureEndEvent
	emStatus = GXSetEnum(m_hDevice, GX_ENUM_EVENT_SELECTOR, GX_ENUM_EVENT_SELECTOR_EXPOSUREEND);
	VERIFY_STATUS_RET(emStatus);
	
	// Enable the EventExposureEnd
	emStatus = GXSetEnum(m_hDevice, GX_ENUM_EVENT_NOTIFICATION, GX_ENUM_EVENT_NOTIFICATION_ON);
	VERIFY_STATUS_RET(emStatus);
	
	return emStatus;
}

//---------------------------------------------------------------------------------
/**
\brief   This is a acquisition callback function that demonstrates how to record the interval that from sending a TriggerSoftware command to receiving a image.
\param   pFrame  

\return  void
*/
//----------------------------------------------------------------------------------
void __stdcall CGxCameraEventsDlg::OnFrameCallbackFun(GX_FRAME_CALLBACK_PARAM *pFrame)
{
	CGxCameraEventsDlg *pDlg = (CGxCameraEventsDlg*)(pFrame->pUserParam);
	double dElapsedtime = 0;
	char   chTemp[8]    = "ms";

	//Stop stopwatch and get the ElapsedTime
	dElapsedtime = pDlg->m_objImageTime.Stop();
	pDlg->PrintInfotoWnd(MsgComposer("The interval that from sending a TriggerSoftware command to receiving a image:%0.2f %s", dElapsedtime, chTemp));

	pDlg->m_bIsTrigValid = true;
}

//---------------------------------------------------------------------------------
/**
\brief    This is a event callback function that demonstrates how to process the ExposureEndEvent and record the interval that from sending a TriggerSoftware command to receiving a ExposureEndEvent
\param    emEventID		
\param	  pUserParam    

\return  void
*/
//----------------------------------------------------------------------------------
void __stdcall CGxCameraEventsDlg::OnFeatureCallbackFun(GX_FEATURE_ID_CMD emFeatureID, void* pUserParam)
{
	CGxCameraEventsDlg *pDlg = (CGxCameraEventsDlg*)pUserParam;
    double dElapsedtime = 0;
	char   chTemp[8]    = "ms";

	if (emFeatureID == GX_INT_EVENT_EXPOSUREEND)
	{
		// Stop timing
		dElapsedtime = pDlg->m_objShutterTime.Stop();
		pDlg->PrintInfotoWnd(MsgComposer("The interval that from sending a TriggerSoftware command to receiving a ExposureEndEvent:%0.2f %s", dElapsedtime, chTemp));
  
	}
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to print the time information in the window
\param  strInfo 
\return void
*/
//----------------------------------------------------------------------------------
void CGxCameraEventsDlg::PrintInfotoWnd(string &strInfo)
{
	int    nTextLen = 0;
	CEdit  *pEdit = (CEdit*)m_pLogWnd;

	if (m_pLogWnd != NULL) 
	{
		nTextLen = pEdit->GetWindowTextLength();	
		if (pEdit->GetLineCount() > 300)
		{
			pEdit->Clear();
			pEdit->SetWindowText(_T(""));
			nTextLen = pEdit->GetWindowTextLength();
		}
		
		pEdit->SetSel(nTextLen, -1);	
		strInfo.append("\r\n");
		pEdit->ReplaceSel((LPCTSTR)strInfo.c_str());
		
		if(nTextLen > 0)
		{
			pEdit->SetSel((nTextLen - pEdit->GetLineCount() + 1), -1);
		}
		else
		{
			pEdit->SetSel(nTextLen, -1);
		}
	}
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to convert the error code to error description
\param  emErrorStatus  
 \return void
*/
//----------------------------------------------------------------------------------
void CGxCameraEventsDlg::ShowErrorString(GX_STATUS emErrorStatus)
{
	char      *pchErrorInfo = NULL;
	size_t    nSize         = 0;
	GX_STATUS emStatus      = GX_STATUS_SUCCESS;
	
	// Get the length of error information and allocate memory.
	emStatus = GXGetLastError(&emErrorStatus, NULL, &nSize);
	pchErrorInfo = new char[nSize];
	if (pchErrorInfo == NULL)
	{
		return;
	}
	
	// Get error information and show them to user 
	emStatus = GXGetLastError(&emErrorStatus, pchErrorInfo, &nSize);
	
	if (emStatus != GX_STATUS_SUCCESS)
	{
		MessageBox("Fail to call GXGetLastError interface!");
	}
	else
	{
		MessageBox((LPCTSTR)pchErrorInfo);
	}
	
	// Release memory 
	if (pchErrorInfo != NULL)
	{
		delete[]pchErrorInfo;
		pchErrorInfo = NULL;
	}
}

//----------------------------------------------------------------------------------
/**
\brief   Update UI

\return  void
*/
//----------------------------------------------------------------------------------
void CGxCameraEventsDlg::UpdateUI()
{
	GetDlgItem(IDC_OPEN_DEVICE)->EnableWindow(!m_bIsOpen);
	GetDlgItem(IDC_START_SNAP)->EnableWindow(m_bIsOpen && (!m_bIsSnap));
	GetDlgItem(IDC_STOP_SNAP)->EnableWindow(m_bIsOpen && m_bIsSnap);
	GetDlgItem(IDC_SEND_SOFT_COMMAND)->EnableWindow(m_bIsOpen && m_bIsSnap);
	GetDlgItem(IDC_CLOSE_DEVICE)->EnableWindow(m_bIsOpen);
}


BOOL CGxCameraEventsDlg::PreTranslateMessage(MSG* pMsg) 
{
	// TODO: Add your specialized code here and/or call the base class
	if ((pMsg->message == WM_KEYDOWN) && (pMsg->wParam == VK_RETURN))   
	{    
		return TRUE;
	}   
	if ((pMsg->message == WM_KEYDOWN) && (pMsg->wParam == VK_ESCAPE))   
	{   
		return  TRUE; 
	}
	
	return CDialog::PreTranslateMessage(pMsg);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to check whether the device support ExposureEndEvent
\param   bSupportExposureEndEvent [out]  

\return  Success: GX_STATUS_SUCCESS, Fail:Else
*/
//----------------------------------------------------------------------------------
GX_STATUS CGxCameraEventsDlg::__IsSupportEvent(bool &bSupportExposureEndEvent)
{
	GX_STATUS emStatus      = GX_STATUS_SUCCESS;

	// Check whether the device supports EventSourceSelector
	emStatus = GXIsImplemented(m_hDevice, GX_ENUM_EVENT_SELECTOR, &m_bImplementEventSelector);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		ShowErrorString(emStatus);
		return emStatus;
	}

	if (m_bImplementEventSelector)
	{
		GX_ENUM_DESCRIPTION *pEnum = NULL;
		size_t    nbufferSize    = 0;
		uint32_t  nEntryNum      = 0;
		int64_t   nEnumValue     = -1;
		int       nCurcel        = 0;
		
		// Get the enumeration entry of the events
		emStatus = GXGetEnumEntryNums(m_hDevice, GX_ENUM_EVENT_SELECTOR, &nEntryNum);
		if (emStatus != GX_STATUS_SUCCESS)
		{
			ShowErrorString(emStatus);
			return emStatus;
		}
		
		// Allocate memory for enumeration
		nbufferSize = nEntryNum * sizeof(GX_ENUM_DESCRIPTION);
		pEnum       = new GX_ENUM_DESCRIPTION[nEntryNum];
		if (pEnum == NULL)
		{
            MessageBox("Fail to allocate buffer !");
			return GX_STATUS_ERROR;
		}
		
		// Get the description of the enumeration
		emStatus = GXGetEnumDescription(m_hDevice, GX_ENUM_EVENT_SELECTOR, pEnum, &nbufferSize);
		if (emStatus != GX_STATUS_SUCCESS)
		{
			if (pEnum != NULL)
			{  
				delete []pEnum;
				pEnum = NULL;
			}
			ShowErrorString(emStatus);
			return emStatus;
		}
		
		// A loop to visit every enumeration entry to check whether it supports ExposureEndEvent 
		for (uint32_t i = 0; i < nEntryNum; i++)
		{	
			if (pEnum[i].nValue == GX_ENUM_EVENT_SELECTOR_EXPOSUREEND)
			{
				bSupportExposureEndEvent = true;
				break;
			}
		}
		
		// Release memory
		if (pEnum != NULL)
		{  
			delete []pEnum;
			pEnum = NULL;
		}
	}

	return emStatus;
}
