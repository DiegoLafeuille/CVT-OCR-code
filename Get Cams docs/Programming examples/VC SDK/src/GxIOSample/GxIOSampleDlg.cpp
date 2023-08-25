// GxIOSampleDlg.cpp : implementation file
//

#include "stdafx.h"
#include "GxIOSample.h"
#include "GxIOSampleDlg.h"

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
// CGxIOSampleDlg dialog

CGxIOSampleDlg::CGxIOSampleDlg(CWnd* pParent /*=NULL*/)
	: CDialog(CGxIOSampleDlg::IDD, pParent)
	, m_bDevOpened(FALSE)
	, m_bIsSnap(FALSE)
	, m_bIsColorFilter(false)
	, m_nPixelColorFilter(GX_COLOR_FILTER_NONE)
	, m_nImageWidth(0)
	, m_nImageHeight(0)
	, m_nPayLoadSize(0)
	, m_nTriggerMode(GX_TRIGGER_MODE_OFF)
	, m_pRawBuffer(NULL)
	, m_pImgBuffer(NULL)
	, m_pBmpInfo(NULL)
	, m_hDevice(NULL)
	, m_bTriggerMode(false)     
	, m_bTriggerActive(false)   
	, m_bTriggerSource(false)   
	, m_bOutputSelector(false)  
	, m_bOutputMode(false)      
	, m_bOutputValue(false)     
	, m_bStrobeSwitch(false) 
	, m_pWnd(NULL)
	, m_hDC(NULL)
	, m_bLineSelector(FALSE)
	, m_bLineMode(FALSE)
	, m_bLineInverter(FALSE)
	, m_bLineSource(FALSE)
	, m_bStrobeSwitchIsUse(TRUE)
{
	//{{AFX_DATA_INIT(CGxIOSampleDlg)
	m_dGainValue = 0.0;
	m_dShutterValue = 0.0;
	//}}AFX_DATA_INIT
	// Note that LoadIcon does not require a subsequent DestroyIcon in Win32
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);

	memset(&m_stShutterFloatRange, 0, sizeof(m_stShutterFloatRange));
	memset(&m_stGainFloatRange, 0, sizeof(m_stGainFloatRange));
    memset(m_chBmpBuf, 0, sizeof(m_chBmpBuf));
}

void CGxIOSampleDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CGxIOSampleDlg)
	DDX_Text(pDX, IDC_EDIT_GAIN_VALUE, m_dGainValue);
	DDX_Text(pDX, IDC_EDIT_SHUTTER_VALUE, m_dShutterValue);
	//}}AFX_DATA_MAP
}

BEGIN_MESSAGE_MAP(CGxIOSampleDlg, CDialog)
	//{{AFX_MSG_MAP(CGxIOSampleDlg)
	ON_WM_QUERYDRAGICON()
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_BN_CLICKED(IDC_BTN_OPEN_DEVICE, OnBnClickedBtnOpenDevice)
	ON_CBN_SELCHANGE(IDC_COMBO_OUTPUT_SELECTOR, OnCbnSelchangeComboOutputSource)
	ON_CBN_SELCHANGE(IDC_COMBO_OUTPUT_MODE, OnCbnSelchangeComboOutputMode)
	ON_CBN_SELCHANGE(IDC_COMBO_OUTPUT_VALUE, OnCbnSelchangeComboOutputValue)
	ON_WM_CLOSE()
	ON_CBN_SELCHANGE(IDC_COMBO_TRIGGER_SOURCE, OnCbnSelchangeComboTriggerSource)
	ON_CBN_SELCHANGE(IDC_COMBO_TRIGGER_MODE, OnCbnSelchangeComboTriggerMode)
	ON_BN_CLICKED(IDC_BTN_SEND_SOFT_TRIGGER, OnBnClickedBtnSendSoftTrigger)
	ON_CBN_SELCHANGE(IDC_COMBO_TRIGGER_ACTIVATION, OnCbnSelchangeComboTriggerActivation)
	ON_BN_CLICKED(IDC_BTN_START_SNAP, OnBnClickedBtnStartSnap)
	ON_CBN_SELCHANGE(IDC_COMBO_STROBE_SWITCH, OnCbnSelchangeComboStrobeSwitch)
	ON_BN_CLICKED(IDC_BTN_STOP_SNAP, OnBnClickedBtnStopSnap)
	ON_BN_CLICKED(IDC_BTN_CLOSE_DEVICE, OnBnClickedBtnCloseDevice)
	ON_EN_KILLFOCUS(IDC_EDIT_SHUTTER_VALUE, OnKillfocusEditShutterValue)
	ON_EN_KILLFOCUS(IDC_EDIT_GAIN_VALUE, OnKillfocusEditGainValue)
	ON_CBN_SELCHANGE(IDC_COMBO_LINE_SELECTOR, OnCbnSelchangeComboLineSelector)
	ON_CBN_SELCHANGE(IDC_COMBO_LINE_MODE, OnCbnSelchangeComboLineMode)
	ON_CBN_SELCHANGE(IDC_COMBO_LINE_INVERTER, OnCbnSelchangeComboLineInverter)
	ON_CBN_SELCHANGE(IDC_COMBO_LINE_SOURCE, OnCbnSelchangeComboLineSource)
	//}}AFX_MSG_MAP

END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CGxIOSampleDlg message handlers

BOOL CGxIOSampleDlg::OnInitDialog()
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
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	// Before using any GxIAPI methods, the GxIAPI must be initialized. 
	emStatus = GXInitLib();
	if (emStatus != GX_STATUS_SUCCESS)
	{
		ShowErrorString(emStatus);
		exit(0);
	}

	UpDateUI();

	// Get the pointer and DC handle for showing image
	m_pWnd       = GetDlgItem(IDC_STATIC_SHOW_WND);
	m_hDC        = ::GetDC(m_pWnd->m_hWnd);

	return TRUE;  // return TRUE  unless you set the focus to a control
}

void CGxIOSampleDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		char strFileName[MAX_PATH] = {'\0'};
		GetModuleFileName(NULL, strFileName, MAX_PATH);
		CFileVersion fv(strFileName);
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

void CGxIOSampleDlg::OnPaint() 
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
HCURSOR CGxIOSampleDlg::OnQueryDragIcon()
{
	return (HCURSOR) m_hIcon;
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to enumerate all camera devices and initialize device.
\return void
*/
//----------------------------------------------------------------------------------
void CGxIOSampleDlg::OnBnClickedBtnOpenDevice() 
{
	// TODO: Add your control notification handler code here
	GX_STATUS     emStatus = GX_STATUS_SUCCESS;
	uint32_t      nDevNum  = 0;
    GX_OPEN_PARAM stOpenParam;
    
	// Enumerate all camera devices 
    stOpenParam.accessMode = GX_ACCESS_EXCLUSIVE;
	stOpenParam.openMode   = GX_OPEN_INDEX;
	stOpenParam.pszContent = "1";

	emStatus = GXUpdateDeviceList(&nDevNum, 1000);
	GX_VERIFY(emStatus);

	// Check if found any device
	if (nDevNum <= 0)
	{
		MessageBox("No devices found!");
		return;
	}

	// If the device is opened then close it to ensure the camera could open again.
	if(m_hDevice != NULL)
	{
		GXCloseDevice(m_hDevice);
		GX_VERIFY(emStatus);
		m_hDevice = NULL;
	}

	//Open the first found device 
	emStatus = GXOpenDevice(&stOpenParam, &m_hDevice);
	GX_VERIFY(emStatus);
	m_bDevOpened = TRUE;

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

	emStatus = InitDevice();
	GX_VERIFY(emStatus);

	emStatus = GetDeviceParam();
	GX_VERIFY(emStatus);

	InitUI();

	UpDateUI();
}

//---------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to initialize the information for setting the LineSelector

\return void
*/
//---------------------------------------------------------------------------------
void CGxIOSampleDlg::UpdateLineRelativeInfo()
{
	InitLineMode();

	InitLineInverter();

	InitLineSource();
}

//---------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to initialize the information for setting the UserOutputSelector

\return void
*/
//---------------------------------------------------------------------------------
void CGxIOSampleDlg::UpdateUserOutputRelativeInfo()
{
	InitOutputValue();
}

//----------------------------------------------------------------------------------
/**
\brief    This function demonstrates how to update UI

\return   void
*/
//----------------------------------------------------------------------------------
void CGxIOSampleDlg::UpDateUI()
{
	GetDlgItem(IDC_BTN_OPEN_DEVICE)->EnableWindow(!m_bDevOpened);
	GetDlgItem(IDC_BTN_CLOSE_DEVICE)->EnableWindow(m_bDevOpened);
	GetDlgItem(IDC_BTN_START_SNAP)->EnableWindow(m_bDevOpened && !m_bIsSnap);
	GetDlgItem(IDC_BTN_STOP_SNAP)->EnableWindow(m_bDevOpened && m_bIsSnap);
	GetDlgItem(IDC_BTN_SEND_SOFT_TRIGGER)->EnableWindow(m_bDevOpened);

	GetDlgItem(IDC_EDIT_GAIN_VALUE)->EnableWindow(m_bDevOpened);
	GetDlgItem(IDC_EDIT_SHUTTER_VALUE)->EnableWindow(m_bDevOpened);

	GetDlgItem(IDC_COMBO_TRIGGER_ACTIVATION)->EnableWindow(m_bDevOpened  && m_bTriggerActive);
	GetDlgItem(IDC_COMBO_TRIGGER_MODE)->EnableWindow(m_bDevOpened && m_bTriggerMode);
	GetDlgItem(IDC_COMBO_TRIGGER_SOURCE)->EnableWindow(m_bDevOpened  && m_bTriggerSource);
	GetDlgItem(IDC_COMBO_OUTPUT_MODE)->EnableWindow(m_bDevOpened  && m_bOutputMode);
	GetDlgItem(IDC_COMBO_OUTPUT_SELECTOR)->EnableWindow(m_bDevOpened  && m_bOutputSelector);
	GetDlgItem(IDC_COMBO_OUTPUT_VALUE)->EnableWindow(m_bDevOpened  && m_bOutputValue);
	GetDlgItem(IDC_COMBO_STROBE_SWITCH)->EnableWindow(m_bDevOpened  && m_bStrobeSwitch && m_bStrobeSwitchIsUse);

	GetDlgItem(IDC_COMBO_LINE_SELECTOR)->EnableWindow(m_bDevOpened && m_bLineSelector);
	GetDlgItem(IDC_COMBO_LINE_MODE)->EnableWindow(m_bDevOpened && m_bLineMode);
	GetDlgItem(IDC_COMBO_LINE_INVERTER)->EnableWindow(m_bDevOpened && m_bLineInverter);
	GetDlgItem(IDC_COMBO_LINE_SOURCE)->EnableWindow(m_bDevOpened && m_bLineSource);
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to convert the error code to error description
\param  emErrorStatus  [in]

\return void
*/
//----------------------------------------------------------------------------------
void CGxIOSampleDlg::ShowErrorString(GX_STATUS emErrorStatus)
{
	char*     pchErrorInfo = NULL;
	size_t    nSize      = 0;
	GX_STATUS emStatus   = GX_STATUS_ERROR;
	
	// Get the length of error information and allocate memory.
	emStatus = GXGetLastError(&emErrorStatus, NULL, &nSize);
	pchErrorInfo = new char[nSize];
	if (NULL == pchErrorInfo)
	{
		return;
	}

	// Get error information and show them to user
	emStatus = GXGetLastError (&emErrorStatus, pchErrorInfo, &nSize);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		MessageBox("Fail to call GXGetLastError interface!");
	}
	else
	{
		MessageBox((LPCTSTR)pchErrorInfo);
	}

	// Release memory 
	if (NULL != pchErrorInfo)
	{
		delete[] pchErrorInfo;
		pchErrorInfo = NULL;
	}
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to change the LineSelector

\return void
*/
//----------------------------------------------------------------------------------
void CGxIOSampleDlg::OnCbnSelchangeComboLineSelector() 
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	int       nIndex   = 0;
	int64_t   nEnumVal = 0;

	// Get the pointer for the LineSelector ComboBox
	CComboBox *pLineSelectorBox = (CComboBox *)GetDlgItem(IDC_COMBO_LINE_SELECTOR);

	// Get current selection
    nIndex   = pLineSelectorBox->GetCurSel();                  
	
	// Get the enumeration entry of current selection
    nEnumVal = (int64_t)pLineSelectorBox->GetItemData(nIndex);

	// Set the value of the enumeration entry 
	emStatus = GXSetEnum(m_hDevice, GX_ENUM_LINE_SELECTOR, nEnumVal);
	GX_VERIFY(emStatus);

	UpdateLineRelativeInfo();
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to change the LineMode		

\return void
*/
//----------------------------------------------------------------------------------
void CGxIOSampleDlg::OnCbnSelchangeComboLineMode() 
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	int       nIndex   = 0;
	int64_t   nEnumVal = 0;

	// Get the pointer for the LineMode ComboBox
	CComboBox *pLineModeBox = (CComboBox *)GetDlgItem(IDC_COMBO_LINE_MODE);

	// Get current selection
	nIndex   = pLineModeBox->GetCurSel();

	// Get the enumeration entry of current selection
	nEnumVal = (int64_t)pLineModeBox->GetItemData(nIndex);

	// Set the value of the enumeration entry 
	emStatus = GXSetEnum(m_hDevice, GX_ENUM_LINE_MODE, nEnumVal);
	GX_VERIFY(emStatus);
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to change the LineInverter

\return void
*/
//----------------------------------------------------------------------------------
void CGxIOSampleDlg::OnCbnSelchangeComboLineInverter() 
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	int       nIndex   = 0;
	bool      bLineInverterValue = TRUE;

	// Get the pointer for the LineInverter ComboBox
	CComboBox *pLineInverter = (CComboBox *)GetDlgItem(IDC_COMBO_LINE_INVERTER);

	// Get current selection
	nIndex   = pLineInverter->GetCurSel();

	// Check and set the value
	if (nIndex == LINE_INVERTER_FALSE)
	{
		bLineInverterValue = FALSE;
	}

	if (nIndex == LINE_INVERTER_TRUE)
	{
		bLineInverterValue = TRUE;
	}

	// Set the value of the enumeration entry 
	emStatus = GXSetBool(m_hDevice, GX_BOOL_LINE_INVERTER, bLineInverterValue);
	GX_VERIFY(emStatus);
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to change the LineSource

\return void
*/
//----------------------------------------------------------------------------------
void CGxIOSampleDlg::OnCbnSelchangeComboLineSource() 
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	int       nIndex   = 0;
	int64_t   nEnumVal = 0;

	// Get the pointer for the LineSource ComboBox
	CComboBox *pLineSource = (CComboBox *)GetDlgItem(IDC_COMBO_LINE_SOURCE);

	// Get current selection
	nIndex   = pLineSource->GetCurSel();

	// Get the enumeration entry of current selection
	nEnumVal = (int64_t)pLineSource->GetItemData(nIndex);

	// Set the value of the enumeration entry 
	emStatus = GXSetEnum(m_hDevice, GX_ENUM_LINE_SOURCE, nEnumVal);
	GX_VERIFY(emStatus);
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to change the UserOutputSelector

\return void
*/
//----------------------------------------------------------------------------------
void CGxIOSampleDlg::OnCbnSelchangeComboOutputSource() 
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	CComboBox *pBox    = (CComboBox *)GetDlgItem(IDC_COMBO_OUTPUT_SELECTOR);

	int      nIndex    = pBox->GetCurSel();                   // Get current selection
	int64_t  nEnumVal  = (int64_t)pBox->GetItemData(nIndex);  // Get the enumeration entry of current selection

	// Set the value of the enumeration entry 
	emStatus = GXSetEnum(m_hDevice, GX_ENUM_USER_OUTPUT_SELECTOR, nEnumVal);
	GX_VERIFY(emStatus);

	UpdateUserOutputRelativeInfo();
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to change the UserOutputMode

\return void
*/
//----------------------------------------------------------------------------------
void CGxIOSampleDlg::OnCbnSelchangeComboOutputMode() 
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	CComboBox *pBox    = (CComboBox *)GetDlgItem(IDC_COMBO_OUTPUT_MODE);

	int      nIndex    = pBox->GetCurSel();                   // Get current selection
	int64_t  nEnumVal  = (int64_t)pBox->GetItemData(nIndex);  // Get the enumeration entry of current selection

	// Set the value of the enumeration entry 
	emStatus = GXSetEnum(m_hDevice, GX_ENUM_USER_OUTPUT_MODE, nEnumVal);
	GX_VERIFY(emStatus);
	
	if (nEnumVal == GX_USER_OUTPUT_MODE_STROBE)
	{
		m_bStrobeSwitchIsUse = TRUE;
	}
	if (nEnumVal == GX_USER_OUTPUT_MODE_USERDEFINED)
	{
		m_bStrobeSwitchIsUse = FALSE;
	}

	UpDateUI();
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to change the UserOutputValue

\return void
*/
//----------------------------------------------------------------------------------
void CGxIOSampleDlg::OnCbnSelchangeComboOutputValue() 
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	CComboBox *pBox    = (CComboBox *)GetDlgItem(IDC_COMBO_OUTPUT_VALUE);

	int      nIndex    = pBox->GetCurSel();                   // Get current selection
	bool     bValue    = false; 
    if (nIndex == OUTPUT_ACTIVATION_FALSE)
    {
		bValue = false;
    }
	else if(nIndex == OUTPUT_ACTIVATION_TRUE)
	{
		bValue = true;
	}

	// Set the value of the enumeration entry 
	emStatus = GXSetBool(m_hDevice, GX_BOOL_USER_OUTPUT_VALUE,bValue);
	GX_VERIFY(emStatus);
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to close the application.

\return void
*/
//----------------------------------------------------------------------------------
void CGxIOSampleDlg::OnClose() 
{
	// TODO: Add your message handler code here and/or call default
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	// If the device is acquiring image then stop it.
	if (m_bIsSnap)
	{
		emStatus = GXSendCommand(m_hDevice, GX_COMMAND_ACQUISITION_STOP);

		// Unregister the CaptureCallback function 
		emStatus = GXUnregisterCaptureCallback(m_hDevice);
		m_bIsSnap = FALSE;

		// Release memory for showing image 
		UnPrepareForShowImg();
	}

	// If the device is opened then close it.
	if (m_bDevOpened)
	{
		emStatus = GXCloseDevice(m_hDevice);
		m_bDevOpened = FALSE;
		m_hDevice    = NULL;
	}

	// Release GxIAPI
	emStatus = GXCloseLib();

	::ReleaseDC(m_pWnd->m_hWnd,m_hDC);	

	CDialog::OnClose();
}

// ---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize device

\return  GX_STATUS_SUCCESS: Success, else: Fail
*/
// ----------------------------------------------------------------------------------
GX_STATUS CGxIOSampleDlg::InitDevice()
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	//Set the continuous frame acquisition mode
	emStatus = GXSetEnum(m_hDevice,GX_ENUM_ACQUISITION_MODE, GX_ACQ_MODE_CONTINUOUS);
	VERIFY_STATUS_RET(emStatus);

	// If the device supports the pixel format of"GX_PIXEL_FORMAT_BAYER_GR8",
	// the following code can be used to set the pixel format to GX_PIXEL_FORMAT_BAYER_GR8:
	// emStatus = GXSetEnum(m_hDevice, GX_ENUM_PIXEL_FORMAT, GX_PIXEL_FORMAT_BAYER_GR8);
	// VERIFY_STATUS_RET(emStatus);

	// If you don't know the device supports which kind of pixel format ,you could use following function to set the pixel format to 8-bit.
	emStatus = SetPixelFormat8bit(); 

	return emStatus;
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to get the parameters of the device

\return GX_STATUS_SUCCESS: Success, else: Fail
*/
//----------------------------------------------------------------------------------
GX_STATUS CGxIOSampleDlg::GetDeviceParam()
{
	GX_STATUS emStatus       = GX_STATUS_ERROR;
	bool      bIsImplemented = false;

	// Get image size
	emStatus = GXGetInt(m_hDevice, GX_INT_PAYLOAD_SIZE, &m_nPayLoadSize);
	VERIFY_STATUS_RET(emStatus);

	// Get image width
	emStatus = GXGetInt(m_hDevice, GX_INT_WIDTH, &m_nImageWidth);
	VERIFY_STATUS_RET(emStatus);

	// Get image height
	emStatus = GXGetInt(m_hDevice, GX_INT_HEIGHT, &m_nImageHeight);
	VERIFY_STATUS_RET(emStatus);

	// Check whether the device supports PixelColorFilter
	emStatus = GXIsImplemented(m_hDevice, GX_ENUM_PIXEL_COLOR_FILTER, &m_bIsColorFilter);
	VERIFY_STATUS_RET(emStatus);

	if(m_bIsColorFilter)
	{
		emStatus = GXGetEnum(m_hDevice, GX_ENUM_PIXEL_COLOR_FILTER, &m_nPixelColorFilter);
	}
    
	return emStatus;
}

// ---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to set the pixel format to 8bit

\return  emStatus GX_STATUS_SUCCESS: Success, else: Fail
*/
// ----------------------------------------------------------------------------------
GX_STATUS CGxIOSampleDlg::SetPixelFormat8bit()
{
	GX_STATUS emStatus    = GX_STATUS_SUCCESS;
	int64_t   nPixelSize  = 0;
	uint32_t  nEnmuEntry  = 0;
	size_t    nBufferSize = 0;
	BOOL      bIs8bit     = TRUE;

	GX_ENUM_DESCRIPTION  *pEnumDescription = NULL;
	GX_ENUM_DESCRIPTION  *pEnumTemp        = NULL;

	// Get the feature PixelSize, this feature indicates the depth of the pixel values in the acquired images in bits per pixel
	emStatus = GXGetEnum(m_hDevice, GX_ENUM_PIXEL_SIZE, &nPixelSize);
	VERIFY_STATUS_RET(emStatus);

	// If the PixelSize is 8bit then return, or set the PixelSize to 8bit.
	if (nPixelSize == GX_PIXEL_SIZE_BPP8)
	{
		return GX_STATUS_SUCCESS;
	}
	else
	{
		// Get the enumeration entry of the pixel format the device supports.
		emStatus = GXGetEnumEntryNums(m_hDevice, GX_ENUM_PIXEL_FORMAT, &nEnmuEntry);
		VERIFY_STATUS_RET(emStatus);

		// Allocate memory for getting the enumeration entry of the pixel format.
		nBufferSize      = nEnmuEntry * sizeof(GX_ENUM_DESCRIPTION);
		pEnumDescription = new GX_ENUM_DESCRIPTION[nEnmuEntry];

		emStatus         = GXGetEnumDescription(m_hDevice, GX_ENUM_PIXEL_FORMAT, pEnumDescription, &nBufferSize);
		if (emStatus != GX_STATUS_SUCCESS)
		{
			if (pEnumDescription != NULL)
			{
				delete pEnumDescription;
				pEnumDescription = NULL;
			}

			return emStatus;
		}

		// A loop to visit every enumeration entry node once to find which enumeration entry node is 8bit and then set pixel format to its value 
		for (uint32_t i = 0; i < nEnmuEntry; i++)
		{
			if ((pEnumDescription[i].nValue & GX_PIXEL_8BIT) == GX_PIXEL_8BIT)
			{
				emStatus = GXSetEnum(m_hDevice, GX_ENUM_PIXEL_FORMAT, pEnumDescription[i].nValue);
				break;
			}
		}	

		// Release memory
		if (pEnumDescription != NULL)
		{
			delete []pEnumDescription;
			pEnumDescription = NULL;
		}
	}

	return emStatus;
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to initialize the TriggerMode Combobox 

\return void
*/
//----------------------------------------------------------------------------------
void CGxIOSampleDlg::InitTriggerModeUI()
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
    
	// Check whether the device supports TriggerMode
	emStatus = GXIsImplemented(m_hDevice, GX_ENUM_TRIGGER_MODE, &m_bTriggerMode);
	GX_VERIFY(emStatus);
	if (!m_bTriggerMode)
	{
		return;
	}

	// Initialize the TriggerMode Combobox 
	InitEnumUI(GX_ENUM_TRIGGER_MODE, (CComboBox *)GetDlgItem(IDC_COMBO_TRIGGER_MODE), m_bTriggerMode);
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to initialize the TriggerSource Combobox

\return void
*/
//----------------------------------------------------------------------------------
void CGxIOSampleDlg::InitTriggerSourceUI()
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	// Check whether the device supports TriggerSource
	emStatus = GXIsImplemented(m_hDevice, GX_ENUM_TRIGGER_SOURCE, &m_bTriggerSource);
	GX_VERIFY(emStatus);
	if (!m_bTriggerSource)
	{
		return;
	}

	// Initialize the TriggerSource Combobox
    InitEnumUI(GX_ENUM_TRIGGER_SOURCE, (CComboBox *)GetDlgItem(IDC_COMBO_TRIGGER_SOURCE),m_bTriggerSource);
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to initialize TriggerActivation Combobox 

\return void
*/
//----------------------------------------------------------------------------------
void CGxIOSampleDlg::InitTriggerActivationUI()
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	// Check whether the device supports TriggerActivation
	emStatus = GXIsImplemented(m_hDevice, GX_ENUM_TRIGGER_ACTIVATION, &m_bTriggerActive);
	GX_VERIFY(emStatus);
	if (!m_bTriggerActive)
	{
		return;
	}

	// Initialize the TriggerActivation Combobox
    InitEnumUI(GX_ENUM_TRIGGER_ACTIVATION, (CComboBox *)GetDlgItem(IDC_COMBO_TRIGGER_ACTIVATION), m_bTriggerActive);
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to initialize StrobeSwitch Combobox 

\return void
*/
//----------------------------------------------------------------------------------
void CGxIOSampleDlg::InitStrobeSwitchUI()
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	// Check whether the device supports StrobeSwitch
	emStatus = GXIsImplemented(m_hDevice, GX_ENUM_STROBE_SWITCH, &m_bStrobeSwitch);
	GX_VERIFY(emStatus);
	if (!m_bStrobeSwitch)
	{
		return;
	}

	// Initialize the StrobeSwitch Combobox
	InitEnumUI(GX_ENUM_STROBE_SWITCH, (CComboBox *)GetDlgItem(IDC_COMBO_STROBE_SWITCH),m_bStrobeSwitch);
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to initialize UserOutputMode Combobox 

\return void
*/
//----------------------------------------------------------------------------------
void CGxIOSampleDlg::InitOutputModeUI()
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	// Check whether the device supports UserOutputMode
	emStatus = GXIsImplemented(m_hDevice, GX_ENUM_USER_OUTPUT_MODE, &m_bOutputMode);
	GX_VERIFY(emStatus);
	if (!m_bOutputMode)
	{
		return;
	}

	// Initialize the UserOutputMode Combobox
    InitEnumUI(GX_ENUM_USER_OUTPUT_MODE, (CComboBox *)GetDlgItem(IDC_COMBO_OUTPUT_MODE), m_bOutputMode);
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to initialize UserOutputSelector Combobox

\return void
*/
//----------------------------------------------------------------------------------
void CGxIOSampleDlg::InitOutputSelectorUI()
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	// Check whether the device supports UserOutputSelector
	emStatus = GXIsImplemented(m_hDevice, GX_ENUM_USER_OUTPUT_SELECTOR, &m_bOutputSelector);
	GX_VERIFY(emStatus);
	if (!m_bOutputSelector)
	{
		return;
	}

	// Initialize the UserOutputSelector Combobox
	InitEnumUI(GX_ENUM_USER_OUTPUT_SELECTOR,(CComboBox *)GetDlgItem(IDC_COMBO_OUTPUT_SELECTOR),m_bOutputSelector);
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to initialize LineSelector Combobox

\return void
*/
//----------------------------------------------------------------------------------
void CGxIOSampleDlg::InitLineSelector()
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	// Check whether the device supports LineSelector
	emStatus = GXIsImplemented(m_hDevice, GX_ENUM_LINE_SELECTOR, &m_bLineSelector);
	GX_VERIFY(emStatus);
	if (!m_bLineSelector)
	{
		return;
	}

	// Initialize the LineSelector Combobox
	InitEnumUI(GX_ENUM_LINE_SELECTOR, (CComboBox *)GetDlgItem(IDC_COMBO_LINE_SELECTOR), m_bLineSelector);
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to initialize LineMode Combobox

\return void
*/
//----------------------------------------------------------------------------------
void CGxIOSampleDlg::InitLineMode()
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	// Check whether the device supports LineMode
	emStatus = GXIsImplemented(m_hDevice, GX_ENUM_LINE_MODE, &m_bLineMode);
	GX_VERIFY(emStatus);
	if (!m_bLineMode)
	{
		return;
	}

	// Initialize the LineMode Combobox
	InitEnumUI(GX_ENUM_LINE_MODE, (CComboBox *)GetDlgItem(IDC_COMBO_LINE_MODE), m_bLineMode);

}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to initialize LineInverter Combobox

\return void
*/
//----------------------------------------------------------------------------------
void CGxIOSampleDlg::InitLineInverter()
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	bool      bTemp    = true;

	CComboBox *pLineInverter = (CComboBox *)GetDlgItem(IDC_COMBO_LINE_INVERTER);

	// Check whether the device supports LineInverter
	emStatus = GXIsImplemented(m_hDevice, GX_BOOL_LINE_INVERTER, &m_bLineInverter);
	GX_VERIFY(emStatus);
	if (!m_bLineInverter)
	{
		return;
	}

	// Initialize the LineInverter
	pLineInverter->ResetContent();
	pLineInverter->SetItemData(pLineInverter->AddString("false"), LINE_INVERTER_FALSE);
	pLineInverter->SetItemData(pLineInverter->AddString("true"), LINE_INVERTER_TRUE);

	// Get the value of LineInverter and update it to UI
	emStatus = GXGetBool(m_hDevice, GX_BOOL_LINE_INVERTER, &bTemp);
	GX_VERIFY(emStatus);
	if (bTemp)
	{
		pLineInverter->SetCurSel(LINE_INVERTER_TRUE);
	}
	else
	{
		pLineInverter->SetCurSel(LINE_INVERTER_FALSE);
	}
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to initialize LineSource Combobox

\return void
*/
//----------------------------------------------------------------------------------
void CGxIOSampleDlg::InitLineSource()
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	// Check whether the device supports LineSource
	emStatus = GXIsImplemented(m_hDevice, GX_ENUM_LINE_SOURCE, &m_bLineSource);
	GX_VERIFY(emStatus);
	if (!m_bLineSource)
	{
		return;
	}

	// Initialize the LineSource Combobox
	InitEnumUI(GX_ENUM_LINE_SOURCE, (CComboBox *)GetDlgItem(IDC_COMBO_LINE_SOURCE), m_bLineSource);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize UI

\return  void
*/
//----------------------------------------------------------------------------------
void CGxIOSampleDlg::InitUI()
{
	GX_STATUS emStatus = GX_STATUS_ERROR;
	int  nValue = 0;
    
    InitOutputValue();

    InitTriggerModeUI();
	
	InitTriggerSourceUI();

	InitTriggerActivationUI();

	InitStrobeSwitchUI();

	InitOutputModeUI();

	InitOutputSelectorUI();
	
	InitShutterUI();

	InitGainUI();

	InitLineSelector();

	InitLineMode();

	InitLineInverter();

	InitLineSource();

	UpdateData(false);
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to initialize UserOutputValue Combobox

\return void
*/
//----------------------------------------------------------------------------------
void CGxIOSampleDlg::InitOutputValue()
{
	CComboBox *pBox           = (CComboBox *)GetDlgItem(IDC_COMBO_OUTPUT_VALUE);
	GX_STATUS  emStatus       = GX_STATUS_SUCCESS;
	bool       bVal           = false;

	// Initialize the TriggerActivation Combobox
	emStatus = GXIsImplemented(m_hDevice,GX_BOOL_USER_OUTPUT_VALUE, &m_bOutputValue);
	GX_VERIFY(emStatus);
	if (!m_bOutputValue)
	{
		return;
	}

	// Initialize the options
	pBox->ResetContent();
	pBox->SetItemData(pBox->AddString("false"), OUTPUT_ACTIVATION_FALSE);
	pBox->SetItemData(pBox->AddString("true"), OUTPUT_ACTIVATION_TRUE);

	// Get the enumeration value and show it to the UI 
	emStatus = GXGetBool(m_hDevice, GX_BOOL_USER_OUTPUT_VALUE, &bVal);
	GX_VERIFY(emStatus);
	if (bVal)
	{
		pBox->SetCurSel(OUTPUT_ACTIVATION_TRUE);
	}
	else
	{
		pBox->SetCurSel(OUTPUT_ACTIVATION_FALSE);
	}
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize the UI of ComboBox                     
\param   emFeatureID    [in]    
\param   pComboBox      [in]   
\param   bIsImplemented [in]    

\return  void
*/
//----------------------------------------------------------------------------------
void CGxIOSampleDlg::InitEnumUI(GX_FEATURE_ID_CMD emFeatureID, CComboBox *pComboBox, bool bIsImplemented)
{
	// Check whether the ComboBox is available
	if ((pComboBox == NULL))
	{
		return;
	}

	GX_ENUM_DESCRIPTION *pEnum       = NULL;
	GX_STATUS emStatus       = GX_STATUS_ERROR;
	size_t    nbufferSize    = 0;
	uint32_t  nEntryNum      = 0;
	int64_t   nEnumValue     = -1;
	int       nCurcel        = 0;

	if (!bIsImplemented)
	{
		return;
	}

	// Get the enumeration entry the device supports.
	emStatus = GXGetEnumEntryNums(m_hDevice, emFeatureID, &nEntryNum);
	GX_VERIFY(emStatus);

	// Allocate memory for getting the description of the enumeration entry
	nbufferSize = nEntryNum * sizeof(GX_ENUM_DESCRIPTION);
	pEnum       = new GX_ENUM_DESCRIPTION[nEntryNum];
	if (pEnum == NULL)
	{
		MessageBox("Fail to allocate memory!");
		return;
	}

	emStatus = GXGetEnumDescription(m_hDevice, emFeatureID, pEnum, &nbufferSize);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		if (pEnum != NULL)
		{  
			delete []pEnum;
			pEnum = NULL;
		}

		return;
	}

	// Get the enumeration value and show it to the UI 
	emStatus = GXGetEnum(m_hDevice, emFeatureID, &nEnumValue);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		if (pEnum != NULL)
		{  
			delete []pEnum;
			pEnum = NULL;
		}

		return;
	}

	// Initialize the options of the ComboBox
	pComboBox->ResetContent();
	for (uint32_t i = 0; i < nEntryNum; i++)
	{	
		pComboBox->SetItemData(pComboBox->AddString(pEnum[i].szSymbolic), (uint32_t)pEnum[i].nValue);
		if (pEnum[i].nValue == nEnumValue)
		{
			nCurcel = i;
		}
	}

    // Show the value to the UI
	pComboBox->SetCurSel(nCurcel);

	// Release memory  
	if (pEnum != NULL)
	{  
		delete []pEnum;
		pEnum = NULL;
	}
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to change the TriggerResource

\return  void
*/
//----------------------------------------------------------------------------------
void CGxIOSampleDlg::OnCbnSelchangeComboTriggerSource()
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	CComboBox *pBox    = (CComboBox *)GetDlgItem(IDC_COMBO_TRIGGER_SOURCE);

	int      nIndex    = pBox->GetCurSel();                   // Get current selection
	int64_t  nEnumVal  = (int64_t)pBox->GetItemData(nIndex);  // Get the enumeration entry of current selection

	emStatus = GXSetEnum(m_hDevice, GX_ENUM_TRIGGER_SOURCE, nEnumVal);
	GX_VERIFY(emStatus);
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to change the TriggerMode

\return void
*/
//----------------------------------------------------------------------------------
void CGxIOSampleDlg::OnCbnSelchangeComboTriggerMode()
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	bool bIsWritable   = TRUE;
	int64_t  nCurrentEnumValue = 0;
	CComboBox *pBox    = (CComboBox *)GetDlgItem(IDC_COMBO_TRIGGER_MODE);

	int      nIndex    = pBox->GetCurSel();                     // Get current selection
	int64_t  nEnumVal  = (int64_t)pBox->GetItemData(nIndex);    // Get the enumeration entry of current selection

	// Check whether the TriggerMode is writable
	emStatus = GXIsWritable(m_hDevice,GX_ENUM_TRIGGER_MODE,&bIsWritable);
	GX_VERIFY(emStatus);

	// Get the enumeration value of the TriggerMode
	emStatus = GXGetEnum(m_hDevice, GX_ENUM_TRIGGER_MODE, &nCurrentEnumValue);
	GX_VERIFY(emStatus);

	if (bIsWritable)
	{
		// Set the value of the enumeration entry 
		emStatus = GXSetEnum(m_hDevice, GX_ENUM_TRIGGER_MODE, nEnumVal);
		GX_VERIFY(emStatus);

		// Update the TriggerMode
		m_nTriggerMode = nEnumVal;
	}
	else
	{
		MessageBox("The TriggerMode is not writable, please try again after stopping acquiring image !");
		if (nCurrentEnumValue != nEnumVal)
		{
			for (int i = 0;i < pBox->GetCount();i++)
			{
				if (nCurrentEnumValue == pBox->GetItemData(i))
				{
					pBox->SetCurSel(i);
					break;
				}
			}
		}
		else
		{
			return;
		}
	}

	UpDateUI();
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to send TriggerSoftware command

\return void
*/
//----------------------------------------------------------------------------------
void CGxIOSampleDlg::OnBnClickedBtnSendSoftTrigger()
{
	// TODO: Add your control notification handler code here
	//Send TriggerSoftware command
	GX_STATUS emStatus = GX_STATUS_ERROR;
	emStatus = GXSendCommand(m_hDevice, GX_COMMAND_TRIGGER_SOFTWARE);

	GX_VERIFY(emStatus);
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to change the TriggerActivation

\return void
*/
//----------------------------------------------------------------------------------
void CGxIOSampleDlg::OnCbnSelchangeComboTriggerActivation()
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	CComboBox *pBox    = (CComboBox *)GetDlgItem(IDC_COMBO_TRIGGER_ACTIVATION);

	int      nIndex    = pBox->GetCurSel();                   // Get current selection
	int64_t  nEnumVal  = (int64_t)pBox->GetItemData(nIndex);  // Get the enumeration entry of current selection

	// Set the value of the enumeration entry 
	emStatus = GXSetEnum(m_hDevice, GX_ENUM_TRIGGER_ACTIVATION, nEnumVal);
	GX_VERIFY(emStatus);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize the ExposureTime control

\return  void
*/
//----------------------------------------------------------------------------------
void CGxIOSampleDlg::InitShutterUI()
{
	CStatic     *pStcShutterShow = (CStatic *)GetDlgItem(IDC_STATIC_SHUTTER);
    CEdit       *pEditShutterVal = (CEdit *)GetDlgItem(IDC_EDIT_SHUTTER_VALUE);

	// Check whether the edit control is available
	if ((pEditShutterVal == NULL) || (pStcShutterShow == NULL))
	{
		return;
	}

    GX_STATUS      emStatus = GX_STATUS_ERROR;
	CString        strTemp  = "";
	double      dShutterVal = 0.0;

	// Get the range of ExposureTime value
	emStatus = GXGetFloatRange(m_hDevice, GX_FLOAT_EXPOSURE_TIME, &m_stShutterFloatRange);
	GX_VERIFY(emStatus);

    strTemp.Format("ExposureTime(%.4f~%.4f)%s",m_stShutterFloatRange.dMin, m_stShutterFloatRange.dMax, m_stShutterFloatRange.szUnit);
	pStcShutterShow->SetWindowText(strTemp);

	// Get current value of ExposureTime and update to the UI
	emStatus = GXGetFloat(m_hDevice, GX_FLOAT_EXPOSURE_TIME, &dShutterVal);
	GX_VERIFY(emStatus);

	m_dShutterValue = dShutterVal;

	UpdateData(FALSE);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize the Gain control 
 
\return  void
*/
//----------------------------------------------------------------------------------
void CGxIOSampleDlg::InitGainUI()
{
	CStatic     *pStcGain     = (CStatic *)GetDlgItem(IDC_STATIC_GAIN);
	CEdit       *pEditGainVal = (CEdit *)GetDlgItem(IDC_EDIT_GAIN_VALUE);

	if ((pStcGain == NULL) || (pEditGainVal == NULL))
	{
		return;
	}
    
	GX_STATUS emStatus = GX_STATUS_ERROR;
	CString   strRange = "";
	double    dGainVal = 0;

	// Get the range of Gain value
	emStatus = GXGetFloatRange(m_hDevice, GX_FLOAT_GAIN, &m_stGainFloatRange);
	GX_VERIFY(emStatus);

	// Show the range in the static control
    strRange.Format("Gain(%.4f~%.4f)%s", m_stGainFloatRange.dMin, m_stGainFloatRange.dMax, m_stGainFloatRange.szUnit);
	pStcGain->SetWindowText(strRange);
	
	// Get current value of Gain and update to the UI
	emStatus = GXGetFloat(m_hDevice, GX_FLOAT_GAIN, &dGainVal);
	GX_VERIFY(emStatus);

	m_dGainValue = dGainVal;
   
	UpdateData(FALSE);
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to start acquisition

\return void
*/
//----------------------------------------------------------------------------------
void CGxIOSampleDlg::OnBnClickedBtnStartSnap()
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_ERROR;

	if (!PrepareForShowImg())
	{
		MessageBox("Fail to allocate resource for showing image!");
		return;
	}

	//Register the CaptureCallback function 
	emStatus = GXRegisterCaptureCallback(m_hDevice, this, OnFrameCallbackFun);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		UnPrepareForShowImg();
		ShowErrorString(emStatus);
		return;
	}

	//Set StreamBufferHandlingMode
	emStatus = GXSetEnum(m_hDevice, GX_DS_ENUM_STREAM_BUFFER_HANDLING_MODE, GX_DS_STREAM_BUFFER_HANDLING_MODE_OLDEST_FIRST);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		UnPrepareForShowImg();
		ShowErrorString(emStatus);
		return;
	}

	//Send AquisitionStart commands
	emStatus = GXSendCommand(m_hDevice, GX_COMMAND_ACQUISITION_START);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		UnPrepareForShowImg();
		ShowErrorString(emStatus);
		return;
	}
	m_bIsSnap = TRUE;

	UpDateUI();
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to allocate memory for showing color image.

\return true:Success false:Fail
*/
//----------------------------------------------------------------------------------
bool  CGxIOSampleDlg::PrepareForShowImg()
{	
	bool bRet = false;

	//Allocate memory for showing images.
	m_pRawBuffer  = new BYTE[(size_t)m_nPayLoadSize];
	if (m_pRawBuffer == NULL)
	{
		bRet = false;
	}

	if (m_bIsColorFilter)   
	{
		// Allocate buffer for showing color image.
		bRet = PrepareForShowColorImg();
	}
	else  
	{   
		// Allocate buffer for showing mono image.
		bRet = PrepareForShowMonoImg();
	}

	// If fail to allocate one of the buffers then release buffers have been allocated. 
	if (!bRet)
	{
		UnPrepareForShowImg();
	}

	return bRet;
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to allocate memory for showing color image.

\return true:Success false:Fail
*/
//----------------------------------------------------------------------------------
bool  CGxIOSampleDlg::PrepareForShowColorImg()
{	
	//--------------------------------------------------------------------
	//----------------------Initialize bitmap header----------------------
	m_pBmpInfo								= (BITMAPINFO *)m_chBmpBuf;
	m_pBmpInfo->bmiHeader.biSize			= sizeof(BITMAPINFOHEADER);
	m_pBmpInfo->bmiHeader.biWidth			= (LONG)m_nImageWidth;
	m_pBmpInfo->bmiHeader.biHeight			= (LONG)m_nImageHeight;	

	m_pBmpInfo->bmiHeader.biPlanes			= 1;
	m_pBmpInfo->bmiHeader.biBitCount		= 24; 
	m_pBmpInfo->bmiHeader.biCompression		= BI_RGB;
	m_pBmpInfo->bmiHeader.biSizeImage		= 0;
	m_pBmpInfo->bmiHeader.biXPelsPerMeter	= 0;
	m_pBmpInfo->bmiHeader.biYPelsPerMeter	= 0;
	m_pBmpInfo->bmiHeader.biClrUsed			= 0;
	m_pBmpInfo->bmiHeader.biClrImportant	= 0;


	//---------------------------------------------------------------------
	//-------------------Allocate buffer for showing image ---------------

	// Allocate buffer for showing converted image.
	m_pImgBuffer = new BYTE[(size_t)(m_nImageWidth * m_nImageHeight * 3)];
	if (m_pImgBuffer == NULL)
	{
		return false;
	}

	return true;
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to allocate memory for showing mono image

\return true:Success false:Fail
*/
//----------------------------------------------------------------------------------
bool  CGxIOSampleDlg::PrepareForShowMonoImg()
{	
	//--------------------------------------------------------------------
	//----------------------Initialize bitmap header----------------------
	m_pBmpInfo								= (BITMAPINFO *)m_chBmpBuf;
	m_pBmpInfo->bmiHeader.biSize			= sizeof(BITMAPINFOHEADER);
	m_pBmpInfo->bmiHeader.biWidth			= (LONG)m_nImageWidth;
	m_pBmpInfo->bmiHeader.biHeight			= (LONG)m_nImageHeight;	

	m_pBmpInfo->bmiHeader.biPlanes			= 1;
	m_pBmpInfo->bmiHeader.biBitCount		= 8;  
	m_pBmpInfo->bmiHeader.biCompression		= BI_RGB;
	m_pBmpInfo->bmiHeader.biSizeImage		= 0;
	m_pBmpInfo->bmiHeader.biXPelsPerMeter	= 0;
	m_pBmpInfo->bmiHeader.biYPelsPerMeter	= 0;
	m_pBmpInfo->bmiHeader.biClrUsed			= 0;
	m_pBmpInfo->bmiHeader.biClrImportant	= 0;

	if (!m_bIsColorFilter)
	{
		// If using mono cameras,you need to initialize color palette first.
		for(int i=0;i<256;i++)
		{
			m_pBmpInfo->bmiColors[i].rgbBlue	 = i;
			m_pBmpInfo->bmiColors[i].rgbGreen	 = i;
			m_pBmpInfo->bmiColors[i].rgbRed		 = i;
			m_pBmpInfo->bmiColors[i].rgbReserved = i;
		}
	}

	//---------------------------------------------------------------------
	//-------------------Allocate buffer for showing image data-------------
	// Allocate memory for showing mono images.
	m_pImgBuffer = new BYTE[(size_t)(m_nImageWidth * m_nImageHeight)];
	if (m_pImgBuffer == NULL)
	{
		return false;
	}

	return true;
}

//---------------------------------------------------------------------------------
/**
\brief   This is a acquisition callback function that demonstrates how to process the acquired image
\param   pFrame      

\return  void
*/
//----------------------------------------------------------------------------------
void __stdcall CGxIOSampleDlg::OnFrameCallbackFun(GX_FRAME_CALLBACK_PARAM* pFrame)
{
	CGxIOSampleDlg *pDlg = (CGxIOSampleDlg*)(pFrame->pUserParam);
	VxUint32 nImageHeight = (VxUint32)pDlg->m_nImageHeight;
	VxUint32 nImageWidth  = (VxUint32)pDlg->m_nImageWidth;

	if (pFrame->status == 0)
	{
		memcpy(pDlg->m_pRawBuffer,pFrame->pImgBuf,pFrame->nImgSize);

		if (pDlg->m_bIsColorFilter)
		{
			// Convert the raw8 image to RGB for showing 
			DxRaw8toRGB24(pDlg->m_pRawBuffer, pDlg->m_pImgBuffer, nImageWidth , nImageHeight, RAW2RGB_NEIGHBOUR,
				           DX_PIXEL_COLOR_FILTER(pDlg->m_nPixelColorFilter), TRUE);
			pDlg->DrawImg(pDlg->m_pImgBuffer);
		}
		else
		{
			// If the acquired image is mono format,you must flip the image data for showing.
			for(VxUint32 i =0;i <nImageHeight;i++)
			{
				memcpy(pDlg->m_pImgBuffer+i*nImageWidth, pDlg->m_pRawBuffer+(nImageHeight-i-1)*nImageWidth,(size_t)nImageWidth);
			}

			pDlg->DrawImg(pDlg->m_pImgBuffer);
		}		
	}
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to draw image in the window
\param   pImageBuffer   [in]  

\return  void
*/
//----------------------------------------------------------------------------------
void CGxIOSampleDlg::DrawImg(BYTE *pImageBuffer)
{
	int  nWndWidth  = 0;
	int  nWndHeight = 0;

	// Initialize the parameters for drawing image.
	RECT objRect;
	m_pWnd->GetClientRect(&objRect);	
	nWndWidth  = objRect.right  - objRect.left;
	nWndHeight = objRect.bottom - objRect.top;

	// You could use the following code to show image. 
	::SetStretchBltMode(m_hDC, COLORONCOLOR);
	::StretchDIBits(m_hDC,
		            0,						
	                0,
		            nWndWidth,
		            nWndHeight,
		            0,
		            0,
		            (int)m_nImageWidth,
		            (int)m_nImageHeight,
		            pImageBuffer,
		            m_pBmpInfo,
		            DIB_RGB_COLORS,
		            SRCCOPY
		            );
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to release memory for showing image.

\return void
*/
//----------------------------------------------------------------------------------
void CGxIOSampleDlg::UnPrepareForShowImg()
{
	if (m_pRawBuffer != NULL)
	{
		delete []m_pRawBuffer;
		m_pRawBuffer = NULL;
	}

	if (m_pImgBuffer != NULL)
	{
		delete []m_pImgBuffer;
		m_pImgBuffer = NULL;
	}
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to change the LineSelector

\return void
*/
//----------------------------------------------------------------------------------
void CGxIOSampleDlg::OnCbnSelchangeComboStrobeSwitch()
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	CComboBox *pBox    = (CComboBox *)GetDlgItem(IDC_COMBO_STROBE_SWITCH);

	int      nIndex    = pBox->GetCurSel();                   // Get current selection
	int64_t  nEnumVal  = (int64_t)pBox->GetItemData(nIndex);  // Get the enumeration entry of current selection

	// Set the value of the enumeration entry 
	emStatus = GXSetEnum(m_hDevice, GX_ENUM_STROBE_SWITCH, nEnumVal);
	GX_VERIFY(emStatus);
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to stop acquisition

\return void
*/
//----------------------------------------------------------------------------------
void CGxIOSampleDlg::OnBnClickedBtnStopSnap()
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	//Send AcquisitionStop command 
	emStatus = GXSendCommand(m_hDevice, GX_COMMAND_ACQUISITION_STOP);
	GX_VERIFY(emStatus);

	//Unregister the CaptureCallback function 
	emStatus = GXUnregisterCaptureCallback(m_hDevice);
	GX_VERIFY(emStatus);
	m_bIsSnap = FALSE;

	UnPrepareForShowImg();

	UpDateUI();
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to close device.

\return void
*/
//----------------------------------------------------------------------------------
void CGxIOSampleDlg::OnBnClickedBtnCloseDevice()
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	// If the device is acquiring image then stop it.
	if (m_bIsSnap)
	{
		emStatus = GXSendCommand(m_hDevice, GX_COMMAND_ACQUISITION_STOP);
		if (emStatus != GX_STATUS_SUCCESS)
		{
			// TODO: Add your error handler code here  
		}

		//Unregister the CaptureCallback function 
		emStatus = GXUnregisterCaptureCallback(m_hDevice);
		if (emStatus != GX_STATUS_SUCCESS)
		{
			// TODO: Add your error handler code here  
		}

		m_bIsSnap = FALSE;

		// Release memory for showing image
		UnPrepareForShowImg();
	}

	emStatus = GXCloseDevice(m_hDevice);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		// TODO: Add your error handler code here  
	}

	m_bDevOpened = FALSE;
	m_hDevice    = NULL;

	UpDateUI();
}

//------------------------------------------------------
/**
\brief   This function demonstrates how to set the ExposureTime

\return  void
*/
//------------------------------------------------------
void CGxIOSampleDlg::OnKillfocusEditShutterValue() 
{
	// TODO: Add your control notification handler code here

	if (m_hDevice == NULL)
	{
		return;
	}

	UpdateData();
	GX_STATUS status = GX_STATUS_SUCCESS;

	if (m_dShutterValue > m_stShutterFloatRange.dMax)
	{
		m_dShutterValue = m_stShutterFloatRange.dMax;
	}

	if (m_dShutterValue < m_stShutterFloatRange.dMin)
	{
		m_dShutterValue = m_stShutterFloatRange.dMin;
	}

	status = GXSetFloat(m_hDevice,GX_FLOAT_EXPOSURE_TIME,m_dShutterValue);
	GX_VERIFY(status);

	UpdateData(FALSE);
}

//------------------------------------------------------
/**
\brief   This function demonstrates how to set the Gain

\return  void
*/
//------------------------------------------------------
void CGxIOSampleDlg::OnKillfocusEditGainValue() 
{
	// TODO: Add your control notification handler code here
	if (m_hDevice == NULL)
	{
		return;
	}

	UpdateData();
	GX_STATUS status = GX_STATUS_SUCCESS;

	if (m_dGainValue > m_stGainFloatRange.dMax)
	{
		m_dGainValue = m_stGainFloatRange.dMax;
	}

	if (m_dGainValue < m_stGainFloatRange.dMin)
	{
		m_dGainValue = m_stGainFloatRange.dMin;
	}

	status = GXSetFloat(m_hDevice,GX_FLOAT_GAIN,m_dGainValue);
	GX_VERIFY(status);

	UpdateData(FALSE);
}

//------------------------------------------------------
/**
\brief   This function demonstrates how to deal with message of "Enter" 
\param   pMsg  

\return  Success:TRUE   Fail:FALSE 
*/
//------------------------------------------------------
BOOL CGxIOSampleDlg::PreTranslateMessage(MSG* pMsg) 
{
	// TODO: Add your specialized code here and/or call the base class
	CWnd *pWnd  = NULL;
	int nCtrlID = 0;              

	if (WM_KEYDOWN == pMsg->message && VK_RETURN == pMsg->wParam)
	{
		pWnd = GetFocus();

		nCtrlID = pWnd->GetDlgCtrlID();

		switch (nCtrlID)
		{
		case IDC_EDIT_SHUTTER_VALUE:
		case IDC_EDIT_GAIN_VALUE:

			SetFocus();

			break;

		default:
			break;
		}

		return TRUE;
	}

	if ((WM_KEYDOWN == pMsg->message) && (VK_ESCAPE == pMsg->wParam))
	{
		return TRUE;
	}

	return CDialog::PreTranslateMessage(pMsg);
}
