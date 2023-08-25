// GxAutoFuncCtrlDlg.cpp : implementation file
//

#include "stdafx.h"
#include "GxAutoFuncCtrl.h"
#include "GxAutoFuncCtrlDlg.h"

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
// CGxAutoFuncCtrlDlg dialog

CGxAutoFuncCtrlDlg::CGxAutoFuncCtrlDlg(CWnd* pParent /*=NULL*/)
	: CDialog(CGxAutoFuncCtrlDlg::IDD, pParent)
	, m_bDevOpened(FALSE)
	, m_bIsSnap(FALSE)
	, m_bIsColorFilter(false)
	, m_bImplementAutoGain(false)
	, m_bImplementAutoShutter(false)
	, m_bImplementLight(false) 
	, m_nGray(0)
	, m_nRoiX(0)
	, m_nRoiW(0)
	, m_nRoiH(0)
	, m_nRoiY(0)
	, m_dAutoGainMin(0)
	, m_dAutoShutterMax(0)
	, m_dAutoGainMax(0)
	, m_dAutoShutterMin(0)
	, m_nImageWidth(0)      
	, m_nImageHeight(0)
	, m_nPayLoadSize(0)
	, m_nPixelColorFilter(0)
	, m_pRawBuffer(NULL)
	, m_pImgBuffer(NULL)
	, m_pBmpInfo(NULL)
	, m_hDevice(NULL)
	, m_emAutoShutterMode(GX_EXPOSURE_AUTO_OFF)
	, m_emAutoGainMode(GX_GAIN_AUTO_OFF)
	, m_pWnd(NULL)
	, m_hDC(NULL)
{
	//{{AFX_DATA_INIT(CGxAutoFuncCtrlDlg)
	m_dShutterValue = 0.0;
	m_dGainValue = 0.0;
	//}}AFX_DATA_INIT
	// Note that LoadIcon does not require a subsequent DestroyIcon in Win32
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);

	memset(&m_stShutterFloatRange, 0, sizeof(m_stShutterFloatRange));
	memset(&m_stGainFloatRange, 0, sizeof(m_stGainFloatRange));
	memset(m_chBmpBuf,0,sizeof(m_chBmpBuf));
}

void CGxAutoFuncCtrlDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CGxAutoFuncCtrlDlg)
	DDX_Text(pDX, IDC_EDIT_GRAY, m_nGray);
	DDX_Text(pDX, IDC_EDIT_ROI_X, m_nRoiX);
	DDX_Text(pDX, IDC_EDIT_ROI_Y, m_nRoiY);
	DDX_Text(pDX, IDC_EDIT_ROI_H, m_nRoiH);
	DDX_Text(pDX, IDC_EDIT_ROI_W, m_nRoiW);
	DDX_Text(pDX, IDC_EDIT_AUTO_GAIN_MIN, m_dAutoGainMin);
	DDX_Text(pDX, IDC_EDIT_AUTO_GAIN_MAX, m_dAutoGainMax);
	DDX_Text(pDX, IDC_EDIT_AUTO_SHUTTER_MIN, m_dAutoShutterMin);
	DDX_Text(pDX, IDC_EDIT_AUTO_SHUTTER_MAX, m_dAutoShutterMax);
	DDX_Text(pDX, IDC_EDIT_CUR_SHUTTER, m_dShutterValue);
	DDX_Text(pDX, IDC_EDIT_CUR_GAIN, m_dGainValue);
	//}}AFX_DATA_MAP
}

BEGIN_MESSAGE_MAP(CGxAutoFuncCtrlDlg, CDialog)
	//{{AFX_MSG_MAP(CGxAutoFuncCtrlDlg)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BTN_OPEN_DEVICE, OnBnClickedBtnOpenDevice)
	ON_BN_CLICKED(IDC_BTN_START_SNAP, OnBnClickedBtnStartSnap)
	ON_BN_CLICKED(IDC_BTN_STOP_SNAP, OnBnClickedBtnStopSnap)
	ON_BN_CLICKED(IDC_BTN_CLOSE_DEVICE, OnBnClickedBtnCloseDevice)
	ON_CBN_SELCHANGE(IDC_COMBO_LIGHT, OnCbnSelchangeComboLight)
	ON_CBN_SELCHANGE(IDC_COMBO_AUTO_GAIN, OnCbnSelchangeComboAutoGain)
	ON_CBN_SELCHANGE(IDC_COMBO_AUTO_SHUTTER, OnCbnSelchangeComboAutoShutter)
	ON_WM_HSCROLL()
	ON_EN_KILLFOCUS(IDC_EDIT_ROI_X, OnKillfocusEditRoiX)
	ON_EN_KILLFOCUS(IDC_EDIT_ROI_Y, OnKillfocusEditRioY)
	ON_EN_KILLFOCUS(IDC_EDIT_ROI_W, OnKillfocusEditRoiW)
	ON_EN_KILLFOCUS(IDC_EDIT_ROI_H, OnKillfocusEditRoiH)
	ON_EN_KILLFOCUS(IDC_EDIT_AUTO_GAIN_MAX, OnKillfocusEditAutoGainMax)
	ON_EN_KILLFOCUS(IDC_EDIT_AUTO_GAIN_MIN, OnKillfocusEditAutoGainMin)
	ON_EN_KILLFOCUS(IDC_EDIT_AUTO_SHUTTER_MAX, OnKillfocusEditAutoShutterMax)
	ON_EN_KILLFOCUS(IDC_EDIT_AUTO_SHUTTER_MIN, OnKillfocusEditAutoShutterMin)
	ON_WM_TIMER()
	ON_WM_CLOSE()
	ON_EN_KILLFOCUS(IDC_EDIT_CUR_SHUTTER, OnKillfocusEditCurShutter)
	ON_EN_KILLFOCUS(IDC_EDIT_CUR_GAIN, OnKillfocusEditCurGain)
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CGxAutoFuncCtrlDlg message handlers

BOOL CGxAutoFuncCtrlDlg::OnInitDialog()
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
	GX_STATUS emStatus = GX_STATUS_ERROR;

	//Before using any GxIAPI methods, the GxIAPI must be initialized. 
	emStatus = GXInitLib();  
	if (emStatus != GX_STATUS_SUCCESS)
	{
		ShowErrorString(emStatus);
		exit(0);
	}
	
	//Get the pointer and DC handle for showing image
	m_pWnd = GetDlgItem(IDC_STATIC_SHOW_WND);
	m_hDC  = ::GetDC(m_pWnd->m_hWnd);

	UpDateUI();
	
	return TRUE;  // return TRUE  unless you set the focus to a control
}

void CGxAutoFuncCtrlDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		char strFileName[MAX_PATH] = {'\0'};
        GetModuleFileName(NULL, strFileName, MAX_PATH);
        CFileVersion fv(strFileName);
        CAboutDlg dlgAbout;		
        dlgAbout.m_strProductVersion = _T("Version:") + fv.GetProductVersion();
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

void CGxAutoFuncCtrlDlg::OnPaint() 
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
HCURSOR CGxAutoFuncCtrlDlg::OnQueryDragIcon()
{
	return (HCURSOR) m_hIcon;
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to enumerate all camera devices and initialize device.

\return  void
*/
//----------------------------------------------------------------------------------
void CGxAutoFuncCtrlDlg::OnBnClickedBtnOpenDevice() 
{
	// TODO: Add your control notification handler code here
	GX_STATUS     emStatus    = GX_STATUS_SUCCESS;
	uint32_t      nDevNum     = 0;
	GX_OPEN_PARAM stOpenParam;

	stOpenParam.accessMode = GX_ACCESS_EXCLUSIVE;
	stOpenParam.openMode   = GX_OPEN_INDEX;
	stOpenParam.pszContent = "1";

	// Enumerate all camera devices 
	emStatus = GXUpdateDeviceList(&nDevNum, 1000);
	GX_VERIFY(emStatus);

	// Check if found any device
	if (nDevNum <= 0)
	{
		MessageBox("No devices found!");
		return;
	}

	// If the device is open then close it to ensure the camera could open again.
	if (m_hDevice != NULL)
	{
		emStatus = GXCloseDevice(m_hDevice);
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
\brief   This function demonstrates how to start acquisition

\return  void
*/
//----------------------------------------------------------------------------------
void CGxAutoFuncCtrlDlg::OnBnClickedBtnStartSnap() 
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_ERROR;

	if(!PrepareForShowImg())
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

	//Send AquisitionStart command
	emStatus = GXSendCommand(m_hDevice, GX_COMMAND_ACQUISITION_START);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		UnPrepareForShowImg();
		ShowErrorString(emStatus);
		return;
	}
	m_bIsSnap = TRUE;

	SetTimer(0,1000,NULL);

	UpDateUI();
}

//---------------------------------------------------------------------------------
/**
\brief   This is a callback function that demonstrates how to convert the image format and draw image
\param   pFrame  

\return  void
*/
//----------------------------------------------------------------------------------
void __stdcall CGxAutoFuncCtrlDlg::OnFrameCallbackFun(GX_FRAME_CALLBACK_PARAM *pFrame)
{
	CGxAutoFuncCtrlDlg *pDlg = (CGxAutoFuncCtrlDlg*)(pFrame->pUserParam);
	VxUint32 nImageHeight = (VxUint32)pDlg->m_nImageHeight;
	VxUint32 nImageWidth  = (VxUint32)pDlg->m_nImageWidth;

	if (pFrame->status == 0)
	{
		memcpy(pDlg->m_pRawBuffer, pFrame->pImgBuf, pFrame->nImgSize);

		if (pDlg->m_bIsColorFilter)
		{
			// If the acquired image is color format,convert it to RGB 
			DxRaw8toRGB24(pDlg->m_pRawBuffer, pDlg->m_pImgBuffer, nImageWidth , nImageHeight, RAW2RGB_NEIGHBOUR,
				DX_PIXEL_COLOR_FILTER(pDlg->m_nPixelColorFilter), TRUE);
		}
		else
		{
			// If the acquired image is mono format,you must flip the image data for showing.
			for(VxUint32 i = 0; i < nImageHeight; i++)
			{
				memcpy(pDlg->m_pImgBuffer + i * nImageWidth, pDlg->m_pRawBuffer + (nImageHeight - i - 1) * nImageWidth, (size_t)nImageWidth);
			}
		}	

		pDlg->DrawImg(pDlg->m_pImgBuffer);
	}
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to stop acquisition

\return  void
*/
//----------------------------------------------------------------------------------
void CGxAutoFuncCtrlDlg::OnBnClickedBtnStopSnap() 
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
    
	// Release memory for showing image 
	UnPrepareForShowImg();
    
	KillTimer(0);

	UpDateUI();
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to close device.

\return  void
*/
//----------------------------------------------------------------------------------
void CGxAutoFuncCtrlDlg::OnBnClickedBtnCloseDevice() 
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

		KillTimer(0);
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

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize the UI of ComboBox 
\param   emFeatureID    [in]   
\param   pComboBox      [in]   
\param   bIsImplemented [in]   

\return  void
*/
//----------------------------------------------------------------------------------
void CGxAutoFuncCtrlDlg::InitEnumUI(GX_FEATURE_ID_CMD emFeatureID, CComboBox *pComboBox, bool bIsImplemented)
{
	// Check whether the ComboBox is available
	if ((pComboBox == NULL) || (!bIsImplemented))
	{
		return;
	}

	GX_ENUM_DESCRIPTION *pEnum = NULL;
	GX_STATUS emStatus       = GX_STATUS_ERROR;
	size_t    nbufferSize    = 0;
	uint32_t  nEntryNum      = 0;
	int64_t   nEnumValue     = -1;
	int       nCurcel        = 0;

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
		ShowErrorString(emStatus);
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
		ShowErrorString(emStatus);
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
\brief   This function demonstrates how to initialize device

\return  void
*/
//----------------------------------------------------------------------------------
GX_STATUS CGxAutoFuncCtrlDlg::InitDevice()
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	//Set the continuous frame acquisition mode
	emStatus = GXSetEnum(m_hDevice,GX_ENUM_ACQUISITION_MODE, GX_ACQ_MODE_CONTINUOUS);
	VERIFY_STATUS_RET(emStatus);

	// Set the TriggerMode to off
	emStatus = GXSetEnum(m_hDevice,GX_ENUM_TRIGGER_MODE, GX_TRIGGER_MODE_OFF);	
	VERIFY_STATUS_RET(emStatus);

	// If the device support the pixel format of"GX_PIXEL_FORMAT_BAYER_GR8",
	// the following code can be used to set the pixel format to GX_PIXEL_FORMAT_BAYER_GR8:
	// emStatus = GXSetEnum(m_hDevice, GX_ENUM_PIXEL_FORMAT, GX_PIXEL_FORMAT_BAYER_GR8);
	// VERIFY_STATUS_RET(emStatus);

	 // If you don't know the device supports which kind of pixel format ,you could use following function to set the pixel format to 8-bit.
	emStatus = SetPixelFormat8bit(); 

	return emStatus;
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize UI

\return  void
*/
//----------------------------------------------------------------------------------
void CGxAutoFuncCtrlDlg::InitUI()
{
	GX_STATUS emStatus = GX_STATUS_ERROR;
	int  nValue = 0;

    InitAutoGainUI();

	InitAutoShutterUI();

	InitLightEnvironmentUI();

    InitGrayUI();

	InitROIUI();

	InitShutterUI();

	InitGainUI();
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize the AutoGain Combobox 

\return  void
*/
//----------------------------------------------------------------------------------
void CGxAutoFuncCtrlDlg::InitAutoGainUI()
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	int64_t nValue = 0;

	// Check whether the device supports AutoGain
	emStatus = GXIsImplemented(m_hDevice, GX_ENUM_GAIN_AUTO, &m_bImplementAutoGain);
	GX_VERIFY(emStatus);

	emStatus = GXGetEnum(m_hDevice,GX_ENUM_GAIN_AUTO,&nValue);
	GX_VERIFY(emStatus);
	m_emAutoGainMode = (GX_GAIN_AUTO_ENTRY)nValue;

	// Initialize the AutoGain Combobox 
	InitEnumUI(GX_ENUM_GAIN_AUTO, (CComboBox *)GetDlgItem(IDC_COMBO_AUTO_GAIN), m_bImplementAutoGain);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize the AutoExposure Combobox 

\return  void
*/
//----------------------------------------------------------------------------------
void CGxAutoFuncCtrlDlg::InitAutoShutterUI()
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	int64_t nValue = 0;

	// Check whether the device supports AutoExposure
	emStatus = GXIsImplemented(m_hDevice, GX_ENUM_EXPOSURE_AUTO, &m_bImplementAutoShutter);
	GX_VERIFY(emStatus);

	emStatus = GXGetEnum(m_hDevice,GX_ENUM_EXPOSURE_AUTO,&nValue);
	GX_VERIFY(emStatus);
	m_emAutoShutterMode = (GX_EXPOSURE_AUTO_ENTRY)nValue;

	// Initialize the AutoExposure Combobox 
	InitEnumUI(GX_ENUM_EXPOSURE_AUTO, (CComboBox *)GetDlgItem(IDC_COMBO_ATUO_SHUTTER), m_bImplementAutoShutter);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize the AA LightEnvironment Combobox

\return  void
*/
//----------------------------------------------------------------------------------
void CGxAutoFuncCtrlDlg::InitLightEnvironmentUI()
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
    
	// Check whether the device supports AA LightEnvironment
	emStatus = GXIsImplemented(m_hDevice, GX_ENUM_AA_LIGHT_ENVIRONMENT, &m_bImplementLight);
	GX_VERIFY(emStatus);

	// Initialize the AA LightEnvironment Combobox 
	InitEnumUI(GX_ENUM_AA_LIGHT_ENVIRONMENT, (CComboBox *)GetDlgItem(IDC_COMBO_LIGHT), m_bImplementLight);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize the ExposureTime control

\return  void
*/
//----------------------------------------------------------------------------------
void CGxAutoFuncCtrlDlg::InitShutterUI()
{
	CStatic     *pStcShutterShow = (CStatic *)GetDlgItem(IDC_STATIC_SHUTTER);
	CEdit       *pEditShutterVal = (CEdit *)GetDlgItem(IDC_EDIT_CUR_SHUTTER);

	// Check whether the edit control is available
	if ((pEditShutterVal == NULL) || (pStcShutterShow == NULL))
	{
		return;
	}

	GX_STATUS      emStatus = GX_STATUS_ERROR;
	CString        strTemp  = "";
	double         dValue   = 0.0;

	// Get the range of ExposureTime value 
	emStatus = GXGetFloatRange(m_hDevice, GX_FLOAT_EXPOSURE_TIME, &m_stShutterFloatRange);
	GX_VERIFY(emStatus);

	strTemp.Format("ExposureTime(%.4f~%.4f)%s", m_stShutterFloatRange.dMin, m_stShutterFloatRange.dMax, m_stShutterFloatRange.szUnit);
	pStcShutterShow->SetWindowText(strTemp);

	// Get current value of ExposureTime and update to the UI
	emStatus = GXGetFloat(m_hDevice, GX_FLOAT_EXPOSURE_TIME, &dValue);
	GX_VERIFY(emStatus);
	m_dShutterValue = dValue;

	// Get current value of AutoExposureTimeMax
	emStatus = GXGetFloat(m_hDevice, GX_FLOAT_AUTO_EXPOSURE_TIME_MAX, &dValue);
	GX_VERIFY(emStatus);
	m_dAutoShutterMax = dValue;
	strTemp.Format("%.4f",m_dAutoShutterMax);
	SetDlgItemText(IDC_EDIT_AUTO_SHUTTER_MAX,strTemp);

	// Get current value of AutoExposureTimeMin
	emStatus = GXGetFloat(m_hDevice, GX_FLOAT_AUTO_EXPOSURE_TIME_MIN, &dValue);
	GX_VERIFY(emStatus);
	m_dAutoShutterMin = dValue;
	strTemp.Format("%.4f",m_dAutoShutterMin);
	SetDlgItemText(IDC_EDIT_AUTO_SHUTTER_MIN,strTemp);

	UpdateData(FALSE);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize the Gain control 
 
\return  void
*/
//----------------------------------------------------------------------------------
void CGxAutoFuncCtrlDlg::InitGainUI()
{
	CStatic     *pStcGain     = (CStatic *)GetDlgItem(IDC_STATIC_GAIN);
	CEdit       *pEditGainVal = (CEdit *)GetDlgItem(IDC_EDIT_CUR_GAIN);

	if ((pStcGain == NULL) || (pEditGainVal == NULL))
	{
		return;
	}
    
	GX_STATUS emStatus = GX_STATUS_ERROR;
	CString   strTemp  = "";
	double    dGainVal = 0.0;

	// Get the range of Gain value
	emStatus = GXGetFloatRange(m_hDevice, GX_FLOAT_GAIN, &m_stGainFloatRange);
	GX_VERIFY(emStatus);

	strTemp.Format("Gain(%.4f~%.4f)%s", m_stGainFloatRange.dMin, m_stGainFloatRange.dMax, m_stGainFloatRange.szUnit);
	pStcGain->SetWindowText(strTemp);
	
	// Get current value of Gain and update to the UI
	emStatus = GXGetFloat(m_hDevice, GX_FLOAT_GAIN, &dGainVal);
	GX_VERIFY(emStatus);
	m_dGainValue= dGainVal;

	// Get current value of AutoGainMax
    emStatus = GXGetFloat(m_hDevice, GX_FLOAT_AUTO_GAIN_MAX, &dGainVal);
	GX_VERIFY(emStatus);
	m_dAutoGainMax = dGainVal;
    strTemp.Format("%.4f",m_dAutoGainMax);
 	SetDlgItemText(IDC_EDIT_AUTO_GAIN_MAX,strTemp);

	// Get current value of AutoGainMin
	emStatus = GXGetFloat(m_hDevice, GX_FLOAT_AUTO_GAIN_MIN, &dGainVal);
	GX_VERIFY(emStatus);
	m_dAutoGainMin = (int)dGainVal;
	strTemp.Format("%.4f",m_dAutoGainMin);
	SetDlgItemText(IDC_EDIT_AUTO_GAIN_MIN,strTemp);
	
	UpdateData(FALSE);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize the ROI control 
 
\return  void
*/
//----------------------------------------------------------------------------------
void CGxAutoFuncCtrlDlg::InitROIUI()
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	int64_t   nValue   = 0;
	
	emStatus = GXGetInt(m_hDevice, GX_INT_AAROI_OFFSETY, &nValue);
	GX_VERIFY(emStatus);
	m_nRoiY = (int)nValue;

	emStatus = GXGetInt(m_hDevice, GX_INT_AAROI_OFFSETX, &nValue);
	GX_VERIFY(emStatus);
	m_nRoiX = (int)nValue;

	emStatus = GXGetInt(m_hDevice, GX_INT_AAROI_WIDTH, &nValue);
	GX_VERIFY(emStatus);
	m_nRoiW = (int)nValue;

	emStatus = GXGetInt(m_hDevice, GX_INT_AAROI_HEIGHT, &nValue);
	GX_VERIFY(emStatus);
	m_nRoiH = (int)nValue;

	// Get the range of the value and update to UI
	UpdateROIRange();

	UpdateData(FALSE);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to get the range of ROI and update to UI
 
\return  void
*/
//----------------------------------------------------------------------------------
void CGxAutoFuncCtrlDlg::UpdateROIRange()
{
	GX_STATUS     emStatus = GX_STATUS_SUCCESS;
	GX_INT_RANGE  nIntRange;
	CString       strTmep = _T("");

	emStatus = GXGetIntRange(m_hDevice, GX_INT_AAROI_WIDTH, &nIntRange);
	GX_VERIFY(emStatus)
    strTmep.Format("AAROIWidth(%I64d~%I64d)",nIntRange.nMin,nIntRange.nMax);
    SetDlgItemText(IDC_STATIC_ROI_W,strTmep);

	emStatus = GXGetIntRange(m_hDevice, GX_INT_AAROI_HEIGHT, &nIntRange);
	GX_VERIFY(emStatus)
	strTmep.Format("AAROIHeight(%I64d~%I64d)",nIntRange.nMin,nIntRange.nMax);
	SetDlgItemText(IDC_STATIC_ROI_H,strTmep);

	emStatus = GXGetIntRange(m_hDevice, GX_INT_AAROI_OFFSETX, &nIntRange);
	GX_VERIFY(emStatus)
	strTmep.Format("AAROIOffsetX(%I64d~%I64d)",nIntRange.nMin,nIntRange.nMax);
	SetDlgItemText(IDC_STATIC_ROI_X,strTmep);

	emStatus = GXGetIntRange(m_hDevice, GX_INT_AAROI_OFFSETY, &nIntRange);
	GX_VERIFY(emStatus)
	strTmep.Format("AA ROIOffsetY(%I64d~%I64d)",nIntRange.nMin,nIntRange.nMax);
	SetDlgItemText(IDC_STATIC_ROI_Y,strTmep);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize the ExpectedGrayValue control 
 
\return  void
*/
//----------------------------------------------------------------------------------
void CGxAutoFuncCtrlDlg::InitGrayUI()
{
	CSliderCtrl *pSliderCtrl  = (CSliderCtrl *)GetDlgItem(IDC_SLIDER_GRAY);
	CStatic     *pStcGray     = (CStatic *)GetDlgItem(IDC_STATIC_GRAY);
	CEdit       *pEditGrayVal = (CEdit *)GetDlgItem(IDC_EDIT_GRAY);

	if ((pSliderCtrl == NULL) || (pStcGray == NULL) || (pEditGrayVal == NULL))
	{
		return;
	}
    
	GX_STATUS emStatus = GX_STATUS_ERROR;
	int32_t   nStep    = 0;
	CString   strRange = "";
	int64_t   nGrayVal = 0;
	GX_INT_RANGE stIntRange;

	// Get the range of ExpectedGrayValue
	emStatus = GXGetIntRange(m_hDevice, GX_INT_GRAY_VALUE, &stIntRange);
	GX_VERIFY(emStatus);
	pSliderCtrl->SetRange((int)stIntRange.nMin, (int)stIntRange.nMax, true);

    strRange.Format("ExpectedGrayValue(%I64d~%I64d)", stIntRange.nMin, stIntRange.nMax);
	pStcGray->SetWindowText(strRange);
	
	// Get current value of ExpectedGrayValue and update to the UI
	emStatus = GXGetInt(m_hDevice, GX_INT_GRAY_VALUE, &nGrayVal);
	GX_VERIFY(emStatus);
	pSliderCtrl->SetPos((int)nGrayVal);
	m_nGray = (int)nGrayVal;
   
	UpdateData(FALSE);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to change the LightEnvironment

\return  void
*/
//----------------------------------------------------------------------------------
void CGxAutoFuncCtrlDlg::OnCbnSelchangeComboLight() 
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	CComboBox *pBox    = (CComboBox *)GetDlgItem(IDC_COMBO_LIGHT);

	int      nIndex    = pBox->GetCurSel();                   // Get current selection
	int64_t  nEnumVal  = (int64_t)pBox->GetItemData(nIndex);  // Get the enumeration entry of current selection


	emStatus = GXSetEnum(m_hDevice, GX_ENUM_AA_LIGHT_ENVIRONMENT, nEnumVal);
	GX_VERIFY(emStatus);
	
	if (((GX_AA_LIGHT_ENVIRMENT_AC50HZ == nEnumVal) 
		|| (GX_AA_LIGHT_ENVIRMENT_AC60HZ == nEnumVal))
		&&((m_emAutoShutterMode != GX_EXPOSURE_AUTO_CONTINUOUS) 
		|| (m_emAutoGainMode != GX_GAIN_AUTO_CONTINUOUS)))
	{
        MessageBox("If you use 50Hz AC or 60Hz AC,\nyou must enable both ExposureAuto and GainAuo, otherwise 2A can not work correctly.","Msg");
	}
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to change the GainAuto

\return  void
*/
//----------------------------------------------------------------------------------
void CGxAutoFuncCtrlDlg::OnCbnSelchangeComboAutoGain() 
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	CComboBox *pBox    = (CComboBox *)GetDlgItem(IDC_COMBO_AUTO_GAIN);

	int      nIndex    = pBox->GetCurSel();                   // Get current selection
	int64_t  nEnumVal  = (int64_t)pBox->GetItemData(nIndex);  // Get the enumeration entry of current selection
    
	emStatus = GXSetEnum(m_hDevice, GX_ENUM_GAIN_AUTO, nEnumVal);
	GX_VERIFY(emStatus);
    m_emAutoGainMode = (GX_GAIN_AUTO_ENTRY)nEnumVal;

 	UpDateUI();
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to change the ExposureAuto

\return  void
*/
//----------------------------------------------------------------------------------
void CGxAutoFuncCtrlDlg::OnCbnSelchangeComboAutoShutter() 
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	CComboBox *pBox    = (CComboBox *)GetDlgItem(IDC_COMBO_ATUO_SHUTTER);

	int      nIndex    = pBox->GetCurSel();                   // Get current selection
	int64_t  nEnumVal  = (int64_t)pBox->GetItemData(nIndex);  // Get the enumeration entry of current selection
    
	emStatus = GXSetEnum(m_hDevice, GX_ENUM_EXPOSURE_AUTO, nEnumVal);
	GX_VERIFY(emStatus);
	m_emAutoShutterMode = (GX_EXPOSURE_AUTO_ENTRY)nEnumVal;
 
 	UpDateUI();
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to set ExpectedGrayValue
\param   nSBCode   
\param   nPos      
\param   pScrollBar  

\return  void
*/
//----------------------------------------------------------------------------------
void CGxAutoFuncCtrlDlg::OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar) 
{
	// TODO: Add your message handler code here and/or call default
	GX_STATUS    emStatus = GX_STATUS_ERROR;
	int64_t      nValue   = 0;
	double       dValue   = 0;
	CString      strTemp  = "";
	CSliderCtrl *pSlider  = (CSliderCtrl *)pScrollBar;

	// Get the current position of the slider
	nValue = pSlider->GetPos();

	switch(pScrollBar->GetDlgCtrlID())
	{
	case IDC_SLIDER_GRAY:  // Set ExpectedGrayValue
		emStatus = GXSetInt(m_hDevice, GX_INT_GRAY_VALUE, nValue);
		m_nGray = (int)nValue;
		break;
	default:
		break;
	}	

	GX_VERIFY(emStatus);
	UpdateData(FALSE);

	CDialog::OnHScroll(nSBCode, nPos, pScrollBar);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to deal with message of "Enter" 
\param   pMsg  

\return  BOOL
*/
//----------------------------------------------------------------------------------
BOOL CGxAutoFuncCtrlDlg::PreTranslateMessage(MSG* pMsg) 
{
	// TODO: Add your specialized code here and/or call the base class
	CWnd  *pWnd     = NULL;
	int   nCtrlID   = 0;

	if ((pMsg->message == WM_KEYDOWN) && (pMsg->wParam == VK_RETURN))   
	{    
		pWnd = GetFocus();

		nCtrlID = pWnd->GetDlgCtrlID();
		switch(nCtrlID)
		{
		case IDC_EDIT_AUTO_GAIN_MAX:
		case IDC_EDIT_AUTO_GAIN_MIN:
		case IDC_EDIT_AUTO_SHUTTER_MAX:
		case IDC_EDIT_AUTO_SHUTTER_MIN:
		case IDC_EDIT_ROI_X:
		case IDC_EDIT_ROI_Y:
		case IDC_EDIT_ROI_W:
		case IDC_EDIT_ROI_H:
		case IDC_EDIT_GRAY: 
		case IDC_EDIT_CUR_SHUTTER:
		case IDC_EDIT_CUR_GAIN:

			SetFocus();

			break;

		default:
			break;
		}

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
\brief   This function demonstrates how to set the AA ROIOffsetX

\return  void
*/
//----------------------------------------------------------------------------------
void CGxAutoFuncCtrlDlg::OnKillfocusEditRoiX() 
{
	// TODO: Add your control notification handler code here
	if (m_hDevice == NULL)
	{
		return;
	}

	GX_STATUS emStatus = GX_STATUS_ERROR;
	UpdateData(TRUE);

	emStatus =GXSetInt(m_hDevice,GX_INT_AAROI_OFFSETX, m_nRoiX);
	GX_VERIFY(emStatus);

	UpdateROIRange();
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to set the AA ROIOffsetY

\return  void
*/
//----------------------------------------------------------------------------------
void CGxAutoFuncCtrlDlg::OnKillfocusEditRioY() 
{
	if (m_hDevice == NULL)
	{
		return;
	}

	GX_STATUS emStatus = GX_STATUS_ERROR;
	UpdateData(TRUE);

	emStatus =GXSetInt(m_hDevice,GX_INT_AAROI_OFFSETY, m_nRoiY);
	GX_VERIFY(emStatus);

	UpdateROIRange();
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to set the AA ROIWidth

\return  void
*/
//----------------------------------------------------------------------------------
void CGxAutoFuncCtrlDlg::OnKillfocusEditRoiW() 
{
	// TODO: Add your control notification handler code here	
	if (m_hDevice == NULL)
	{
		return;
	}

	GX_STATUS emStatus = GX_STATUS_ERROR;
	UpdateData(TRUE);

	emStatus =GXSetInt(m_hDevice,GX_INT_AAROI_WIDTH, m_nRoiW);
	GX_VERIFY(emStatus);

	UpdateROIRange();
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to set the AA ROIHeight

\return  void
*/
//----------------------------------------------------------------------------------
void CGxAutoFuncCtrlDlg::OnKillfocusEditRoiH() 
{
	// TODO: Add your control notification handler code here
	if (m_hDevice == NULL)
	{
		return;
	}

	GX_STATUS emStatus = GX_STATUS_ERROR;
	UpdateData(TRUE);

	emStatus =GXSetInt(m_hDevice,GX_INT_AAROI_HEIGHT, m_nRoiH);
	GX_VERIFY(emStatus);

	UpdateROIRange();
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to set the AutoGainMax

\return  void
*/
//----------------------------------------------------------------------------------
void CGxAutoFuncCtrlDlg::OnKillfocusEditAutoGainMax() 
{
	// TODO: Add your control notification handler code here
	if (m_hDevice == NULL)
	{
		return;
	}

	GX_STATUS emStatus = GX_STATUS_ERROR;
	UpdateData(TRUE);

	if (m_dAutoGainMax > m_stGainFloatRange.dMax)
	{
		m_dAutoGainMax = m_stGainFloatRange.dMax;
	}

	emStatus = GXSetFloat(m_hDevice, GX_FLOAT_AUTO_GAIN_MAX, m_dAutoGainMax);
	GX_VERIFY(emStatus);

	UpdateData(FALSE);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to set the AutoGainMin

\return  void
*/
//----------------------------------------------------------------------------------
void CGxAutoFuncCtrlDlg::OnKillfocusEditAutoGainMin() 
{
	// TODO: Add your control notification handler code here
	if (m_hDevice == NULL)
	{
		return;
	}

    GX_STATUS emStatus = GX_STATUS_ERROR;
	UpdateData(TRUE);

	if (m_dAutoGainMin < m_stGainFloatRange.dMin)
	{
		m_dAutoGainMin = m_stGainFloatRange.dMin;
	}

	emStatus = GXSetFloat(m_hDevice, GX_FLOAT_AUTO_GAIN_MIN, m_dAutoGainMin);
	GX_VERIFY(emStatus);

	UpdateData(FALSE);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to set the AutoExposureTimeMax

\return  void
*/
//----------------------------------------------------------------------------------
void CGxAutoFuncCtrlDlg::OnKillfocusEditAutoShutterMax() 
{
	// TODO: Add your control notification handler code here
	if (m_hDevice == NULL)
	{
		return;
	}

	GX_STATUS emStatus = GX_STATUS_ERROR;
	UpdateData(TRUE);

	if (m_dAutoShutterMax > m_stShutterFloatRange.dMax)
	{
		m_dAutoShutterMax = m_stShutterFloatRange.dMax;
	}

	emStatus =GXSetFloat(m_hDevice,GX_FLOAT_AUTO_EXPOSURE_TIME_MAX, m_dAutoShutterMax);
	GX_VERIFY(emStatus);

	UpdateData(FALSE);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to set the AutoExposureTimeMin

\return  void
*/
//----------------------------------------------------------------------------------
void CGxAutoFuncCtrlDlg::OnKillfocusEditAutoShutterMin() 
{
	// TODO: Add your control notification handler code here
	if (m_hDevice == NULL)
	{
		return;
	}

	GX_STATUS emStatus = GX_STATUS_ERROR;
	UpdateData(TRUE);

	if (m_dAutoShutterMin < m_stShutterFloatRange.dMin)
	{
		m_dAutoShutterMin = m_stShutterFloatRange.dMin;
	}

	emStatus =GXSetFloat(m_hDevice,GX_FLOAT_AUTO_EXPOSURE_TIME_MIN, m_dAutoShutterMin);
	GX_VERIFY(emStatus);

	UpdateData(FALSE);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to update current Gain and Exposure to UI
 
\return  void
*/
//---------------------------------------------------------------------------------
void CGxAutoFuncCtrlDlg::RefreshCurValue()
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	CString   strTemp  = "";
	double    dValue = 0;

	if (m_emAutoShutterMode != GX_EXPOSURE_AUTO_OFF)
	{
		//Check whether the ExposureAuto is ONCE and then update UI 
		if (m_emAutoShutterMode == GX_EXPOSURE_AUTO_ONCE)
		{
			int64_t  nShutterEnumVal = 0;  
			
			//The pointer for ExposureAuto ComboBox
			CComboBox *pAutoShutterBox    = (CComboBox *)GetDlgItem(IDC_COMBO_ATUO_SHUTTER);
			
			//Get the enumeration entry of ExposureAuto
			emStatus = GXGetEnum(m_hDevice, GX_ENUM_EXPOSURE_AUTO, &nShutterEnumVal);
			if (emStatus != GX_STATUS_SUCCESS)
			{
				return;
			}
			
			m_emAutoShutterMode = (GX_EXPOSURE_AUTO_ENTRY)nShutterEnumVal;
			
			// Check whether the ExposureAuto is OFF and then update UI
			if (m_emAutoShutterMode == GX_EXPOSURE_AUTO_OFF)
			{
				for (int i = 0;i < pAutoShutterBox->GetCount();i++)
				{
					if ((int64_t)pAutoShutterBox->GetItemData(i) == GX_EXPOSURE_AUTO_OFF)
					{
						pAutoShutterBox->SetCurSel(i);
						break;
					}
				}
				
				UpDateUI();
			}
		}

		//Get current ExposureTime value and update to UI
		emStatus = GXGetFloat(m_hDevice, GX_FLOAT_EXPOSURE_TIME, &dValue);
		m_dShutterValue = dValue;
		strTemp.Format("%.4f", m_dShutterValue);
		SetDlgItemText(IDC_EDIT_CUR_SHUTTER, strTemp);
	}
	
	if (m_emAutoGainMode != GX_GAIN_AUTO_OFF)
	{
		//Check whether the GainAuto is ONCE and then update UI 
		if (m_emAutoGainMode == GX_GAIN_AUTO_ONCE)
		{
			int64_t  nGainEnumVal = 0;   
			
			//The pointer for GainAuto ComboBox
			CComboBox *pAutoGainBox    = (CComboBox *)GetDlgItem(IDC_COMBO_AUTO_GAIN);
			
			//Get the enumeration entry of GainAuto
			emStatus = GXGetEnum(m_hDevice, GX_ENUM_GAIN_AUTO, &nGainEnumVal);
			if (emStatus != GX_STATUS_SUCCESS)
			{
				return;
			}
			
			m_emAutoGainMode = (GX_GAIN_AUTO_ENTRY)nGainEnumVal;
			
			// Check whether the GainAuto is OFF and then update UI
			if (m_emAutoGainMode == GX_GAIN_AUTO_OFF)
			{
				for (int i = 0;i < pAutoGainBox->GetCount();i++)
				{
					if ((int64_t)pAutoGainBox->GetItemData(i) == GX_GAIN_AUTO_OFF)
					{
						pAutoGainBox->SetCurSel(i);
						break;
					}
				}
				
				UpDateUI();		
			}
		}

		// Get current Gain value and update to UI
		emStatus  = GXGetFloat(m_hDevice, GX_FLOAT_GAIN, &dValue);
		m_dGainValue = dValue;
		strTemp.Format("%.4f", m_dGainValue);
		SetDlgItemText(IDC_EDIT_CUR_GAIN, strTemp);
	}	
}

//-----------------------------------------------------------
/**
\brief   This is a callback function of timer for updating UI.

\return  void
*/
//-----------------------------------------------------------
void CGxAutoFuncCtrlDlg::OnTimer(UINT_PTR nIDEvent) 
{
	// TODO: Add your message handler code here and/or call default	

	RefreshCurValue();

	CDialog::OnTimer(nIDEvent);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to close the application.

\return  void
*/
//----------------------------------------------------------------------------------
void CGxAutoFuncCtrlDlg::OnClose() 
{
	// TODO: Add your message handler code here and/or call default
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	
	KillTimer(0);
	
	// If the device is acquiring image then stop it.
	if (m_bIsSnap)
	{
		// Send AcquisitionStop command
		emStatus = GXSendCommand(m_hDevice, GX_COMMAND_ACQUISITION_STOP);
		
		// Unregister the CaptureCallback function 
		emStatus = GXUnregisterCaptureCallback(m_hDevice);
		m_bIsSnap = FALSE;
		
		// Release memory for showing image 
		UnPrepareForShowImg();
	}
	
	::ReleaseDC(m_pWnd->m_hWnd, m_hDC);
	
	// If the device is opened then close it.
	if (m_bDevOpened)
	{
		emStatus = GXCloseDevice(m_hDevice);
		m_bDevOpened = FALSE;
		m_hDevice    = NULL;
	}
	
	// Release GxIAPI
	emStatus = GXCloseLib();

	CDialog::OnClose();
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to update UI

\return  void
*/
//----------------------------------------------------------------------------------
void CGxAutoFuncCtrlDlg::UpDateUI()
{
	GetDlgItem(IDC_BTN_OPEN_DEVICE)->EnableWindow(!m_bDevOpened);
	GetDlgItem(IDC_BTN_CLOSE_DEVICE)->EnableWindow(m_bDevOpened);
	GetDlgItem(IDC_BTN_START_SNAP)->EnableWindow(m_bDevOpened && !m_bIsSnap);
	GetDlgItem(IDC_BTN_STOP_SNAP)->EnableWindow(m_bDevOpened && m_bIsSnap);

	GetDlgItem(IDC_COMBO_AUTO_GAIN)->EnableWindow(m_bDevOpened);
	GetDlgItem(IDC_COMBO_AUTO_SHUTTER)->EnableWindow(m_bDevOpened);
	GetDlgItem(IDC_COMBO_LIGHT)->EnableWindow(m_bDevOpened && m_bImplementLight);
	GetDlgItem(IDC_SLIDER_GRAY)->EnableWindow(m_bDevOpened);

	GetDlgItem(IDC_EDIT_ROI_Y)->EnableWindow(m_bDevOpened);
	GetDlgItem(IDC_EDIT_ROI_H)->EnableWindow(m_bDevOpened);
	GetDlgItem(IDC_EDIT_ROI_X)->EnableWindow(m_bDevOpened);
	GetDlgItem(IDC_EDIT_ROI_W)->EnableWindow(m_bDevOpened);


	GetDlgItem(IDC_EDIT_AUTO_GAIN_MAX)->EnableWindow(m_bDevOpened);
	GetDlgItem(IDC_EDIT_AUTO_GAIN_MIN)->EnableWindow(m_bDevOpened);
	GetDlgItem(IDC_EDIT_CUR_GAIN)->EnableWindow(m_bDevOpened && (m_emAutoGainMode == GX_GAIN_AUTO_OFF));

	GetDlgItem(IDC_EDIT_AUTO_SHUTTER_MIN)->EnableWindow(m_bDevOpened);
	GetDlgItem(IDC_EDIT_AUTO_SHUTTER_MAX)->EnableWindow(m_bDevOpened);
	GetDlgItem(IDC_EDIT_CUR_SHUTTER)->EnableWindow(m_bDevOpened && (m_emAutoShutterMode == GX_EXPOSURE_AUTO_OFF));
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to convert the error code to error description
\param  emErrorStatus  [in] 

\return void
*/
//----------------------------------------------------------------------------------
void CGxAutoFuncCtrlDlg::ShowErrorString(GX_STATUS emErrorStatus)
{
	char*     pchErrorInfo = NULL;
	size_t    nSize        = 0;
	GX_STATUS emStatus     = GX_STATUS_ERROR;
	
	// Get the length of error information and allocate memory.
	emStatus = GXGetLastError(&emErrorStatus, NULL, &nSize);
	pchErrorInfo = new char[nSize];
	if (NULL == pchErrorInfo)
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
	if (NULL != pchErrorInfo)
	{
		delete[] pchErrorInfo;
		pchErrorInfo = NULL;
	}
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to get the parameters of the device

\return GX_STATUS_SUCCESS:Success, Else: Fail
*/
//----------------------------------------------------------------------------------
GX_STATUS CGxAutoFuncCtrlDlg::GetDeviceParam()
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
GX_STATUS CGxAutoFuncCtrlDlg::SetPixelFormat8bit()
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
				delete []pEnumDescription;
				pEnumDescription = NULL;
			}

			return emStatus;
		}

		// A loop to visit every enumeration entry node once to find which enumeration entry node is 8bit and then set pixel format to its value 
		for (uint32_t i = 0; i<nEnmuEntry; i++)
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
\brief  This function demonstrates how to allocate memory for showing image.

\return true:Success false:Fail
*/
//----------------------------------------------------------------------------------
bool  CGxAutoFuncCtrlDlg::PrepareForShowImg()
{	
	bool bRet = true;

	//Allocate memory for getting image
	m_pRawBuffer = new BYTE[(size_t)m_nPayLoadSize];
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
bool  CGxAutoFuncCtrlDlg::PrepareForShowColorImg()
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
	//-------------------Allocate buffer for showing image ----------------

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
\brief  This function demonstrates how to allocate memory for showing mono image.

\return true:Success false:Fail
*/
//----------------------------------------------------------------------------------
bool  CGxAutoFuncCtrlDlg::PrepareForShowMonoImg()
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
		for(int i = 0; i < 256; i++)
		{
			m_pBmpInfo->bmiColors[i].rgbBlue	 = i;
			m_pBmpInfo->bmiColors[i].rgbGreen	 = i;
			m_pBmpInfo->bmiColors[i].rgbRed		 = i;
			m_pBmpInfo->bmiColors[i].rgbReserved = i;
		}
	}

	//---------------------------------------------------------------------
	//-------------------Allocate buffer for showing image data------------
	// Allocate memory for showing mono images.
	m_pImgBuffer = new BYTE[(size_t)(m_nImageWidth * m_nImageHeight)];
	if (m_pImgBuffer == NULL)
	{
		return false;
	}

	return true;
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to release memory for showing image.

\return void
*/
//----------------------------------------------------------------------------------
void CGxAutoFuncCtrlDlg::UnPrepareForShowImg()
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

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to draw image in the window

\return  void
*/
//----------------------------------------------------------------------------------
void CGxAutoFuncCtrlDlg::DrawImg(BYTE *pImageBuffer)
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

//---------------------------------------------------
/**
\brief   This function demonstrates how to set the ExposureTime

\return  void
*/
//---------------------------------------------------
void CGxAutoFuncCtrlDlg::OnKillfocusEditCurShutter() 
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

	status = GXSetFloat(m_hDevice,GX_FLOAT_EXPOSURE_TIME, m_dShutterValue);
	GX_VERIFY(status);

	UpdateData(FALSE);
}

//-------------------------------------------------------
/**
\brief    This function demonstrates how to set the Gain

\return   void
*/
//-------------------------------------------------------
void CGxAutoFuncCtrlDlg::OnKillfocusEditCurGain() 
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
