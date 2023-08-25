// GxCameraEventsDlg.cpp : implementation file
//----------------------------------------------------------------------------------
/**
\brief  CGxCameraEventsDlg class implementation
*/
//----------------------------------------------------------------------------------

#include "stdafx.h"
#include "GxCameraEvents.h"
#include "GxCameraEventsDlg.h"
#include "FileVersion.h"

using namespace std;

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// CAboutDlg dialog used for App About
class CAboutDlg : public CDialog
{
public:
	CAboutDlg();

    // Dialog Data
	enum { IDD = IDD_ABOUTBOX };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

protected:
	DECLARE_MESSAGE_MAP()
public:
	CString m_strProductVersion;
	CString m_strLegalCopyRight;
};

CAboutDlg::CAboutDlg() : CDialog(CAboutDlg::IDD)
, m_strProductVersion(_T(""))
, m_strLegalCopyRight(_T(""))
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_STATIC_PRODUCTVERSION, m_strProductVersion);
	DDX_Text(pDX, IDC_STATIC_LEGALCOPYRIGHT, m_strLegalCopyRight);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialog)
END_MESSAGE_MAP()

// CGxCameraEventsDlg dialog
CGxCameraEventsDlg::CGxCameraEventsDlg(CWnd* pParent /*=NULL*/)
	: CDialog(CGxCameraEventsDlg::IDD, pParent)
	, m_bIsOpen(false)
	, m_bIsSnap(false)
	, m_hCB(NULL)
	, m_pLogWnd(NULL)
	, m_pCaptureEventHandle(NULL)
	, m_pFeatureEventHandle(NULL)
	, m_bIsTrigValid(true)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);

}
void CGxCameraEventsDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CGxCameraEventsDlg, CDialog)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	//}}AFX_MSG_MAP
	ON_BN_CLICKED(IDC_OPEN_DEVICE, &CGxCameraEventsDlg::OnBnClickedOpenDevice)
	ON_BN_CLICKED(IDC_START_SNAP, &CGxCameraEventsDlg::OnBnClickedStartSnap)
	ON_BN_CLICKED(IDC_STOP_SNAP, &CGxCameraEventsDlg::OnBnClickedStopSnap)
	ON_BN_CLICKED(IDC_CLOSE_DEVICE, &CGxCameraEventsDlg::OnBnClickedCloseDevice)
	ON_BN_CLICKED(IDC_SEND_SOFT_COMMAND, &CGxCameraEventsDlg::OnBnClickedSendSoftCommand)
	ON_WM_CLOSE()
END_MESSAGE_MAP()

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
	try
	{
		// Before using any GxIAPICPP methods, the GxIAPICPP must be initialized. 
		IGXFactory::GetInstance().Init();

		// The pointer for showing the time to the window
		m_pLogWnd =GetDlgItem(IDC_EDIT_TIME_LOG);

		// Update UI
		__UpdateUI();
	    
		m_pCaptureEventHandle = new CSampleCaptureEventHandle();
		m_pFeatureEventHandle = new CSampleFeatureEventHandle();
	}
	catch (CGalaxyException &e)
	{
		MessageBox(e.what());
		return FALSE;
		
	}
	catch (std::exception &e)
	{
		MessageBox(e.what());
		return FALSE;

	}

	// return TRUE  unless you set the focus to a control
	return TRUE;  
}

void CGxCameraEventsDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		char chFileName[MAX_PATH] = {'\0'};
		GetModuleFileName(NULL,chFileName,MAX_PATH);
		CFileVersion fv(chFileName);
		CAboutDlg dlgAbout;
		dlgAbout.m_strProductVersion = _T("Version:") + fv.GetProductVersion();
		dlgAbout.m_strLegalCopyRight =fv.GetLegalCopyright();
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

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

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

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CGxCameraEventsDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

void CGxCameraEventsDlg::OnBnClickedOpenDevice()
{
	// TODO: Add your control notification handler code here
	bool bIsDeviceOpen              = false;       ///< The flag indicates whether the device has been opened
	bool bIsStreamOpen              = false;       ///< The flag indicates whether the device stream has been opened
	bool m_bSupportExposureEndEvent = false;       ///< The flag indicates whether the device supports ExposureEndEvent

    try
    {
		// Enumerate all camera devices 
		GxIAPICPP::gxdeviceinfo_vector vectorDeviceInfo;
		IGXFactory::GetInstance().UpdateDeviceList(1000,vectorDeviceInfo);
		if (vectorDeviceInfo.size() <= 0)
		{
			MessageBox("No device found!");
			return;
		}

		// Open Device
		m_objDevicePtr =IGXFactory::GetInstance().OpenDeviceBySN(vectorDeviceInfo[0].GetSN(),GX_ACCESS_EXCLUSIVE);
		bIsDeviceOpen = true;

		// Get Feature Control 
		m_objFeatureControlPtr = m_objDevicePtr->GetRemoteFeatureControl();

		// Check whether the device supports ExposureEndEvent
		m_bSupportExposureEndEvent = __IsSupportExposureEnd();

		if (!m_bSupportExposureEndEvent)
		{
			throw exception("This sample is invalid because of the camera doesn't support ExposureEndEvent !");
		}

		// Get the number of device stream channels
		uint32_t nStreamCount =m_objDevicePtr->GetStreamCount();

		// Open device stream
		if (nStreamCount > 0)
		{
			m_objStreamPtr = m_objDevicePtr->OpenStream(0);
			m_objStreamFeatureControlPtr = m_objStreamPtr->GetFeatureControl();
			bIsStreamOpen  = true;
		}
		else
		{
			throw exception("No device stream found!");
		}

        //It is recommended that the user set the camera's stream channel packet length value 
        //according to the current network environment after turning on 
        //the network camera to improve the collection performance of the network camera. 
        //For the setting method, refer to the following code.
        GX_DEVICE_CLASS_LIST objDeviceClass = m_objDevicePtr->GetDeviceInfo().GetDeviceClass();
        if(GX_DEVICE_CLASS_GEV == objDeviceClass)
        {
            // Determine whether the device supports the stream channel packet function.
            if(true == m_objFeatureControlPtr->IsImplemented("GevSCPSPacketSize"))
            {
                // Get the optimal packet length value of the current network environment
                int nPacketSize = m_objStreamPtr->GetOptimalPacketSize();
                // Set the optimal packet length value to the stream channel packet length of the current device.
                m_objFeatureControlPtr->GetIntFeature("GevSCPSPacketSize")->SetValue(nPacketSize);
            } 
        }

		// Initialize camera parameters
		__InitParam();

		m_bIsOpen = true ;

		// Update UI
		__UpdateUI();
    }
    catch (CGalaxyException &e)
    {
		// Check whether the device stream has been opened
		if (bIsStreamOpen)
		{
			m_objStreamPtr->Close();
		}

		// Check whether the device has been opened
		if (bIsDeviceOpen)
		{
			m_objDevicePtr->Close();
		}

    	MessageBox(e.what());
		return;
    }
	catch (std::exception& e)
	{
		// Check whether the device stream has been opened
		if (bIsStreamOpen)
		{
			m_objStreamPtr->Close();
		}

		// Check whether the device has been opened
		if (bIsDeviceOpen)
		{
			m_objDevicePtr->Close();
		}

		MessageBox(e.what());
		return;
	}
}

void CGxCameraEventsDlg::OnBnClickedStartSnap()
{
	// TODO: Add your control notification handler code here
	try
	{
		// Set StreamBufferHandlingMode
		m_objStreamFeatureControlPtr->GetEnumFeature("StreamBufferHandlingMode")->SetValue("OldestFirst");

		// Register ExposureEndEvent callback function
		m_hCB = m_objFeatureControlPtr->RegisterFeatureCallback("EventExposureEnd",
			                                                     m_pFeatureEventHandle, this);
		
		// Register the CaptureCallback function
		m_objStreamPtr->RegisterCaptureCallback(m_pCaptureEventHandle, this);

		// Start stream channel acquisition
		m_objStreamPtr->StartGrab();

		// Send AcquisitionStart command
		m_objFeatureControlPtr->GetCommandFeature("AcquisitionStart")->Execute();
		m_bIsSnap = true;
		m_bIsTrigValid = true;

		// Update UI
		__UpdateUI();

	}
	catch (CGalaxyException &e)
	{
		MessageBox(e.what());
		return;	
	}
	catch (std::exception &e)
	{
		MessageBox(e.what());
		return;	
	}
}

void CGxCameraEventsDlg::OnBnClickedStopSnap()
{
	// TODO: Add your control notification handler code here
	try
	{
		// Send AcquisitionStop command
		m_objFeatureControlPtr->GetCommandFeature("AcquisitionStop")->Execute();

		// Stop stream channel acquisition
		m_objStreamPtr->StopGrab();

		// Unregister the ExposureEndEvent callback function
		m_objFeatureControlPtr->UnregisterFeatureCallback(m_hCB);

		// Unregister the CaptureCallback function
		m_objStreamPtr->UnregisterCaptureCallback();
		m_bIsSnap = false;

		// Update UI
		__UpdateUI();

	}
	catch (CGalaxyException &e)
	{
		MessageBox(e.what());
		return;	
	}
	catch (std::exception &e)
	{
		MessageBox(e.what());
		return;	
	}
}

void CGxCameraEventsDlg::OnBnClickedCloseDevice()
{
	// TODO: Add your control notification handler code here
	try
	{
		// Check whether the device has been stoped acquisition
		if (m_bIsSnap)
		{
			// Send AcquisitionStop command
			m_objFeatureControlPtr->GetCommandFeature("AcquisitionStop")->Execute();

			// Stop Stream acquisition
			m_objStreamPtr->StopGrab();

			// Unregister the ExposureEndEvent callback function
			m_objStreamPtr->UnregisterCaptureCallback();

			// Unregister the CaptureCallback function
			m_objFeatureControlPtr->UnregisterFeatureCallback(m_hCB);

		}
	}
	catch(CGalaxyException)
	{
		//do noting
	}

	try
	{
		// Close stream
		m_objStreamPtr->Close();

		// Close device
		m_objDevicePtr->Close();
	}
	catch(CGalaxyException)
	{
		//do noting
	}

	m_bIsOpen = false;
	m_bIsSnap = false;

	// Update UI
	__UpdateUI();
}

void CGxCameraEventsDlg::OnBnClickedSendSoftCommand()
{
	// TODO: Add your control notification handler code here
	try
	{
		// If you click the trigger button again while the trigger callback is executing, the click will be displayed
		if (!m_bIsTrigValid)
		{
			return;
		}
		else
		{
			m_bIsTrigValid = false;
		}

		SYSTEMTIME    system;
		CString       strTime = "";      ///< Time string

		// Get the time for sending TriggerSoftware command 
		GetLocalTime(&system);
		strTime.Format("%02dh:%0.2dm:%0.2ds:%03dms ",
			           system.wHour,system.wMinute,system.wSecond,system.wMilliseconds);

		// Print time
		CString strInformation = "\r\nThe time of sending TriggerSoftware command:" + strTime;
		std::string strPrintInformation = strInformation.GetBuffer(0);
		PrintInfotoWnd(strPrintInformation);		

		// Start timing for receiving image
		m_objImageTime.Start();

		// Start timing for recording the process of ExposureEndEvent
		m_objShutterTime.Start();

		// Send TriggerSoftware command
		m_objFeatureControlPtr->GetCommandFeature("TriggerSoftware")->Execute();
	}
	catch (CGalaxyException &e)
	{
		MessageBox(e.what());
		m_bIsTrigValid = true;
		return;	
	}
	catch (std::exception &e)
	{
		MessageBox(e.what());
		m_bIsTrigValid = true;
		return;	
	}
}

void CGxCameraEventsDlg::OnClose()
{
	// TODO: Add your message handler code here and/or call default
	try
	{
		// Check whether the device has been stoped acquisition
		if (m_bIsSnap)
		{
			// Send AcquisitionStop command
			m_objFeatureControlPtr->GetCommandFeature("AcquisitionStop")->Execute();

			// Stop Stream acquisition
			m_objStreamPtr->StopGrab();

			// Unregister the ExposureEndEvent callback function
			m_objFeatureControlPtr->UnregisterFeatureCallback(m_hCB);

			// Unregister the CaptureCallback function
			m_objStreamPtr->UnregisterCaptureCallback();

			m_bIsSnap = false;

		}
	}
	catch(CGalaxyException)
	{
		//do noting
	}

	try
	{
		// Check whether the device has been closed
		if (m_bIsOpen)
		{
			// Close stream
			m_objStreamPtr->Close();

			// Close device
			m_objDevicePtr->Close();

			m_bIsOpen = false;
		}
	}
	catch(CGalaxyException)
	{
		//do noting
	}

	try
	{
		// Release GxIAPICPP
		IGXFactory::GetInstance().Uninit();

	}
	catch(CGalaxyException)
	{
		//do noting
	}

	if (m_pCaptureEventHandle != NULL)
	{
		delete m_pCaptureEventHandle;
		m_pCaptureEventHandle = NULL;
	}

	if (m_pFeatureEventHandle != NULL)
	{
		delete m_pFeatureEventHandle;
		m_pFeatureEventHandle = NULL;
	}

	CDialog::OnClose();
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize device 

\return  emStatus GX_STATUS_SUCCESS: Success, Else: Fail
*/
//----------------------------------------------------------------------------------
void CGxCameraEventsDlg::__InitParam()
{
	// Set the continuous frame acquisition mode
	m_objFeatureControlPtr->GetEnumFeature("AcquisitionMode")->SetValue("Continuous");

	// Set the TriggerMode on
	m_objFeatureControlPtr->GetEnumFeature("TriggerMode")->SetValue("On");

	// Set the TriggerSource to SoftTrigger
	m_objFeatureControlPtr->GetEnumFeature("TriggerSource")->SetValue("Software");

	// Select the ExposureEndEvent
	m_objFeatureControlPtr->GetEnumFeature("EventSelector")->SetValue("ExposureEnd");

	// Enable the EventExposureEnd
	m_objFeatureControlPtr->GetEnumFeature("EventNotification")->SetValue("On");   

}

//----------------------------------------------------------------------------------
/**
\brief   Check whether the device supports ExposureEndEvent

\return   true: supports ExposureEndEvent   false: not supports ExposureEndEvent
*/
//----------------------------------------------------------------------------------
bool CGxCameraEventsDlg::__IsSupportExposureEnd()
{ 
	bool bIsSupportExposureEnd = false;   ///< The flag indicates whether the device supports ExposureEndEvent

	// enumerate Event container
	GxIAPICPP::gxstring_vector vectorEnumEvent;
	CEnumFeaturePointer pEnumFeature = m_objFeatureControlPtr->GetEnumFeature("EventSelector");
	vectorEnumEvent = pEnumFeature->GetEnumEntryList();
	if (vectorEnumEvent.size() <= 0)
	{
		return bIsSupportExposureEnd;
	}

	// whether the device supports ExposureEndEvent
	for (uint32_t i=0; i<vectorEnumEvent.size(); i++)
	{
		if (vectorEnumEvent[i] == "ExposureEnd")
		{
			bIsSupportExposureEnd = true;
			return bIsSupportExposureEnd;
		}
	}

	return bIsSupportExposureEnd;
}

//----------------------------------------------------------------------------------
/**
\brief  Update UI

\return void
*/
//----------------------------------------------------------------------------------
void CGxCameraEventsDlg::__UpdateUI()
{
	GetDlgItem(IDC_OPEN_DEVICE)->EnableWindow(!m_bIsOpen);
	GetDlgItem(IDC_START_SNAP)->EnableWindow(m_bIsOpen && (!m_bIsSnap));
	GetDlgItem(IDC_STOP_SNAP)->EnableWindow(m_bIsOpen && m_bIsSnap);
	GetDlgItem(IDC_CLOSE_DEVICE)->EnableWindow(m_bIsOpen);
	GetDlgItem(IDC_SEND_SOFT_COMMAND)->EnableWindow(m_bIsOpen && m_bIsSnap);
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
	try
	{
		int      nTextLen = 0;
		CEdit*   pEdit    = (CEdit*)m_pLogWnd;      ///< Point to the edit box handle

		if (pEdit != NULL)
		{
			nTextLen = pEdit->GetWindowTextLength();
			if (pEdit->GetLineCount() > 300)
			{
				pEdit->Clear();
				pEdit->SetWindowText(_T(""));
				nTextLen = pEdit->GetWindowTextLength();
			}

			pEdit->SetSel(nTextLen,-1);
			strInfo.append("\r\n");
			pEdit->ReplaceSel((LPCTSTR)strInfo.c_str());

			if (nTextLen >0)
			{
				pEdit->SetSel((nTextLen - pEdit->GetWindowTextLength() + 1), -1);
			}
			else
			{
				pEdit->SetSel(nTextLen,-1);
			}

		}
	}
	catch (std::exception& e)
	{
		MessageBox(e.what());
		return;
	}
	
}

BOOL CGxCameraEventsDlg::PreTranslateMessage(MSG* pMsg)
{
	CWnd  *pWnd   = NULL;
	int   nCtrlID = 0;             

	// Check whether the keyboard enter the message
	if ((pMsg->message == WM_KEYDOWN) && (pMsg->wParam == VK_RETURN))   
	{ 
		// Get control focus
		pWnd = GetFocus();

		nCtrlID = pWnd->GetDlgCtrlID();

		// Lose focus
		SetFocus();

		return TRUE;
	}   
	if ((pMsg->message == WM_KEYDOWN) && (pMsg->wParam == VK_ESCAPE))  
	{   
		return  TRUE; 
	}

	return CDialog::PreTranslateMessage(pMsg);
}
