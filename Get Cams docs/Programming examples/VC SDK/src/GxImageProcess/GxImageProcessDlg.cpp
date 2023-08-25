// GxImageProcessDlg.cpp : implementation file
//

#include "StdAfx.h"
#include "GxImageProcess.h"
#include "GxImageProcessDlg.h"
#include "string"


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
// CGxImageProcessDlg dialog

CGxImageProcessDlg::CGxImageProcessDlg(CWnd* pParent /*=NULL*/)
	: CDialog(CGxImageProcessDlg::IDD, pParent)
{
	//{{AFX_DATA_INIT(CGxImageProcessDlg)
	//}}AFX_DATA_INIT
	// Note that LoadIcon does not require a subsequent DestroyIcon in Win32
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);

	m_hDevice             = NULL;            // The handle for device 
	m_pBmpInfo            = NULL;	         // The pointer for showing image
	m_pBufferRaw          = NULL;            // The pointer for the raw data of image
	m_pBufferImage        = NULL;            // RGB Image data for display
	m_pBufferMono         = NULL;            // Mono Image processing cache for rollover
	m_nImageWidth         = 0;               // ImageWidth	
	m_nImageHeight        = 0;               // ImageHeight
	m_nPayLoadSize        = 0;               // PayLoadSize   
	m_nPixelColorFilter   = 0;               // PixelColorFilter
	m_bColorFilter        = false;           // The flag indicates whether the device supports PixelColorFilter 
	m_bAwbLampHouse       = false;           // The flag indicates whether the device supports AwbLampHouse 
	m_bColorCorrection    = false;           // The flag indicates whether the device supports ColorCorrection
	m_bIsOpenCC           = true;            // The flag indicates whether the ColorCorrection is opened
	m_bIsOpen             = false;           // The flag indicates whether the first device has been opened
	m_bIsSnap             = false;           // The flag indicates whether image acquisition has started
	m_nGetColorCorrection = 0;               // The acquired value of ColorCorrection 
	m_nColorCorrection    = 0;               // The value of ColorCorrection 
	m_nContrast           = 0;               // The value of Contrast 
	m_dGamma              = 0;               // The value of Gamma 
	m_nLightness          = 0;               // The value of Lightness 
	m_nSaturation         = 0;               // The value of Saturation 
    m_nPreColorCorrection = 0;               // The value of ColorCorrection before operation
	m_nPreSaturation      = 0;               // The value of Saturation before operation 
	m_nPreContrast        = 0;               // The value of Contrast before operation
	m_nPreLightness       = 0;               // The value of Lightness before operation
	m_dPreGamma           = 0;               // The value of Gamma before operation
	m_fSharpen            = 0;               // The value of Sharpen 
	m_bUpData             = true;            // The flag indicates whether the showing CPU information is updated 
	m_hWndHandle          = NULL;            // The handle for showing image 
    m_hDC                 = NULL;            // The handle for drawing image DC 
	m_nAutoWhiteBalance   = 0;               // The value of AutoWhiteBalance
    m_bEnableColorCorrect = false;           // Color correction enables flag bit
    m_bEnableGamma        = false;           // Gamma enabled flag bit
    m_bEnableSharpness    = false;           // Sharpen enable flag bit
    memset(m_chBmpBuf,0,sizeof(m_chBmpBuf));

	m_bAccelerate = false;
	m_bColorCorrection = false;
	m_bDefectivePixelCorrect = false;
	m_bSharpness = false;
	m_bDenoise = false;
	m_bImageProcess = false;

	m_pBaseinfo = NULL;

	// The color image processing function sets the pointer to initialize in the structure
	m_objImageProcess.parrCC  = NULL;
	m_objImageProcess.pProLut = NULL;
	
	// The black and white image processing function sets the pointer to initialize in the structure
	m_objMonoImgProc.pProLut = NULL;
}

void CGxImageProcessDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CGxImageProcessDlg)
	DDX_Control(pDX, IDC_SLIDER_GAMMA, m_sliderGamma);
	DDX_Control(pDX, IDC_SLIDER_CONTRAST, m_sliderContrast);
	DDX_Control(pDX, IDC_SLIDER_LIGHTNESS, m_sliderLightness);
	DDX_Control(pDX, IDC_SLIDER_SATURATION, m_sliderSaturation);
	DDX_Control(pDX, IDC_SLIDER_SHARPEN, m_sliderSharpen);
	//}}AFX_DATA_MAP
	DDX_Control(pDX, IDC_COMBO_ChooseDevice, m_comboChooseDevice);
}

BEGIN_MESSAGE_MAP(CGxImageProcessDlg, CDialog)
	//{{AFX_MSG_MAP(CGxImageProcessDlg)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_OPEN_DEVICE, OnOpenDevice)
	ON_BN_CLICKED(IDC_START_SNAP, OnStartSnap)
	ON_BN_CLICKED(IDC_STOP_SNAP, OnStopSnap)
	ON_BN_CLICKED(IDC_CLOSE_DEVICE, OnCloseDevice)
	ON_CBN_SELCHANGE(IDC_COMBO_AWB, OnSelchangeComboAwb)
	ON_CBN_SELCHANGE(IDC_COMBO_AWB_LIGHT, OnSelchangeComboAwbLight)
	ON_WM_CLOSE()
	ON_WM_HSCROLL()
	ON_BN_CLICKED(IDC_OPEN_BAD_PIXEL_CORRECT, OnOpenBadPixelCorrect)
	ON_BN_CLICKED(IDC_OPEN_DENOISE, OnOpenDenoise)
	ON_BN_CLICKED(IDC_OPEN_ACCELERATE, OnOpenAccelerate)
	ON_BN_CLICKED(IDC_OPEN_SHARPEN, OnOpenSharpen)
	ON_BN_CLICKED(IDC_OPEN_COLOR_CORRECT, OnOpenColorCorrect)
	//}}AFX_MSG_MAP
	ON_WM_TIMER()
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CGxImageProcessDlg message handlers

BOOL CGxImageProcessDlg::OnInitDialog()
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
	
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	// Before using any GxIAPI methods, the GxIAPI must be initialized. 
	emStatus = GXInitLib();
	if (emStatus != GX_STATUS_SUCCESS)
	{
		ShowErrorString(emStatus);
		exit(0); 
	}

	// Get the pointer and DC handle for showing image
	GetDlgItem(IDC_STATIC_SHOW_FRAME, &m_hWndHandle);
    m_hDC  = ::GetDC(m_hWndHandle); 
	::GetClientRect(m_hWndHandle, &m_objRect);

	// Initial drop-down list control
	__InitCombo();

	UpdateUI();
	
	return TRUE;  // return TRUE  unless you set the focus to a control
}

//----------------------------------------------------------------
/**
\brief  Initial drop-down list control
*/
//----------------------------------------------------------------
void CGxImageProcessDlg::__InitCombo()
{
	GX_STATUS      emStatus   = GX_STATUS_SUCCESS;
	uint32_t       nDeviceNum = 0;

	// Enumerate the number of devices
	emStatus = GXUpdateDeviceList(&nDeviceNum, 1000);
	GX_VERIFY(emStatus);

	// Determine the current number of connected devices
	if (nDeviceNum <= 0)
	{
		MessageBox("Did not enumerate to the device, please insert the camera and restart the program!");
		m_bEnableOpen = false;
		return;
	}
	else
	{
		// Get basic information about all devices
		m_pBaseinfo = new (std::nothrow) GX_DEVICE_BASE_INFO[nDeviceNum];
		if(NULL == m_pBaseinfo)
		{
			MessageBox("Failed to allocate buffer!");
			return;
		}
		
		size_t nSize = nDeviceNum * sizeof(GX_DEVICE_BASE_INFO);
		emStatus = GXGetAllDeviceBaseInfo(m_pBaseinfo, &nSize);
		GX_VERIFY(emStatus);
		

		GX_DEVICE_BASE_INFO *pTmp = m_pBaseinfo;
		for (unsigned int i = 0; i < nDeviceNum; i++)
		{
			m_comboChooseDevice.AddString(pTmp->szDisplayName);
			m_vecDevInfo.push_back(*pTmp);
			pTmp++;
		}

		// The default drop-down list selects the first camera
		m_comboChooseDevice.SetCurSel(0);
		m_bEnableOpen = true;
	}
}

void CGxImageProcessDlg::OnSysCommand(UINT nID, LPARAM lParam)
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

void CGxImageProcessDlg::OnPaint() 
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
HCURSOR CGxImageProcessDlg::OnQueryDragIcon()
{
	return (HCURSOR) m_hIcon;
}

void CGxImageProcessDlg::OnClose() 
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	
	// If the device is acquiring image then stop it.
	if (m_bIsSnap)
	{
		// Send AcquisitionStop commands
		emStatus = GXSendCommand(m_hDevice, GX_COMMAND_ACQUISITION_STOP);
		
		// Unregister the CaptureCallback function 
		emStatus = GXUnregisterCaptureCallback(m_hDevice);
		
		// Release memory for showing image 
		UnPrepareForShowImage();	
		m_bIsSnap = false;
	}
	
	if (m_bIsOpen)
	{
		// Close the camera
		emStatus = GXCloseDevice(m_hDevice);
		m_bIsOpen = false;
		m_hDevice = NULL;

		// Release memory for image process
		UnInitImageProcessStruct();
	}

	// Release GxIAPI
	emStatus = GXCloseLib();

	if (NULL != m_pBaseinfo)
	{
		delete []m_pBaseinfo;
		m_pBaseinfo = NULL;
	}

	// Release the display window handle
	::ReleaseDC(m_hWndHandle, m_hDC);

	CDialog::OnClose();
}

//----------------------------------------------------------------
/**
\brief  Open the device response function
*/
//----------------------------------------------------------------
void CGxImageProcessDlg::OnOpenDevice() 
{
	GX_STATUS      emStatus   = GX_STATUS_SUCCESS;
	bool           bReturn    = false;
	GX_OPEN_PARAM  stOpenParam;
	bool	       bImplementPacketSize = false;
	uint32_t       unPacketSize		  = 0;

	
	// If the device is opened then close it to ensure the camera could open again.
	if (m_hDevice != NULL)
	{
		emStatus = GXCloseDevice(m_hDevice);
		GX_VERIFY(emStatus);
		m_hDevice = NULL;
	}

	// Open the first found device 
	int nSel = m_comboChooseDevice.GetCurSel();
	if (nSel < 0)
	{
		MessageBox("Please insert the camera!");
		return;
	}

	stOpenParam.accessMode = GX_ACCESS_EXCLUSIVE;
	stOpenParam.openMode   = GX_OPEN_SN;
	stOpenParam.pszContent = m_vecDevInfo[nSel].szSN;

	emStatus = GXOpenDevice(&stOpenParam, &m_hDevice);
	GX_VERIFY(emStatus);

    // It is suggested that users set the camera's packet size according to the current network environment after opening the network camera.
    // To improve the acquisition performance of network cameras, the following code is used for reference.
	
	// Determine whether the device supports the channel packet function
	emStatus = GXIsImplemented(m_hDevice, GX_INT_GEV_PACKETSIZE, &bImplementPacketSize);
	if (GX_STATUS_SUCCESS != emStatus)
	{
		    GXCloseDevice(m_hDevice);
		    ShowErrorString(emStatus); 
		    return ;
	}

	if (bImplementPacketSize)
	{
		// Gets the optimal packet length value for the current network environment
		emStatus = GXGetOptimalPacketSize(m_hDevice,&unPacketSize);
		if (GX_STATUS_SUCCESS != emStatus)
	    {
		    GXCloseDevice(m_hDevice);
		    ShowErrorString(emStatus); 
		    return ;
	    }

		// Set the optimal packet length value to the channel packet length value of the current device
		emStatus = GXSetInt(m_hDevice, GX_INT_GEV_PACKETSIZE, unPacketSize);
		if (GX_STATUS_SUCCESS != emStatus)
	    {
		    GXCloseDevice(m_hDevice);
		    ShowErrorString(emStatus); 
		    return ;
	    }
	}
	
	// Sharpening, algorithm acceleration, bad point correction, image noise reduction is not enabled by default
	m_bSharpness = false;
	m_bAccelerate = false;
	m_bDefectivePixelCorrect = false;
	m_bDenoise = false;

	// initialization parameter
	emStatus = InitParam();
	if (GX_STATUS_SUCCESS != emStatus)
	{
		GXCloseDevice(m_hDevice);
		ShowErrorString(emStatus); 
		return ;
	}
	
	// Initializes the image processing functional structure
	bReturn = InitImageProcessStruct();
	if (!bReturn)
	{
		GXCloseDevice(m_hDevice);
		MessageBox("Failed to initialize image processing function structure!");
		return;
	}

	// Initializes the UI interface control
	InitUI();
	m_bIsOpen = true;	
	m_bEnableOpen = false;

	//Set the timer to update the interface
	SetTimer(0, 1000, NULL);
	
	// Update UI interface
	UpdateUI();
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize device 

\return  emStatus GX_STATUS_SUCCESS:Success, else:Fail
*/
//----------------------------------------------------------------------------------
GX_STATUS CGxImageProcessDlg::InitParam()
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	
	// Set bayer format to 0
	m_nPixelColorFilter = 0;

	// The collection mode is set to continuous collection
	emStatus = GXSetEnum(m_hDevice, GX_ENUM_ACQUISITION_MODE, GX_ACQ_MODE_CONTINUOUS);
	VERIFY_STATUS_RET(emStatus);

	// Set the TriggerMode off
	emStatus = GXSetEnum(m_hDevice,GX_ENUM_TRIGGER_MODE, GX_TRIGGER_MODE_OFF);	
	VERIFY_STATUS_RET(emStatus);
	
	// If the device support the pixel format of"GX_PIXEL_FORMAT_BAYER_GR8",
	// the following code can be used to set the pixel format to GX_PIXEL_FORMAT_BAYER_GR8:
	// emStatus = GXSetEnum(m_hDevice, GX_ENUM_PIXEL_FORMAT, GX_PIXEL_FORMAT_BAYER_GR8);
	// VERIFY_STATUS_RET(emStatus);
	
	// If you don't know the device supports which kind of pixel format ,you could use following function to set the pixel format to 8-bit.
	emStatus = SetPixelFormat8bit(); 
	VERIFY_STATUS_RET(emStatus);
	
	// Get PayLoadSize
	emStatus = GXGetInt(m_hDevice, GX_INT_PAYLOAD_SIZE, &m_nPayLoadSize);
	VERIFY_STATUS_RET(emStatus);
	
	// Get ImageWidth
	emStatus = GXGetInt(m_hDevice, GX_INT_WIDTH, &m_nImageWidth);
	VERIFY_STATUS_RET(emStatus);
	
	// Get ImageHeight
	emStatus = GXGetInt(m_hDevice, GX_INT_HEIGHT, &m_nImageHeight);
	VERIFY_STATUS_RET(emStatus);
	
	// Check whether the device supports PixelColorFilter
	emStatus = GXIsImplemented(m_hDevice, GX_ENUM_PIXEL_COLOR_FILTER, &m_bColorFilter);
	VERIFY_STATUS_RET(emStatus);
	
	// Get PixelColorFilter
	if(m_bColorFilter)
	{
		emStatus = GXGetEnum(m_hDevice, GX_ENUM_PIXEL_COLOR_FILTER, &m_nPixelColorFilter);
		VERIFY_STATUS_RET(emStatus);
	}
	
	// Check whether the device supports BalanceRatioSelector 
	emStatus = GXIsImplemented(m_hDevice, GX_ENUM_AWB_LAMP_HOUSE, &m_bAwbLampHouse);
	VERIFY_STATUS_RET(emStatus);

	emStatus = GetImageParam();
	
	return emStatus;
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to get image parameters

\return  emStatus GX_STATUS_SUCCESS: Success, else: Fail
*/
//----------------------------------------------------------------------------------
GX_STATUS CGxImageProcessDlg::GetImageParam()
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	bool      bIsImplemented = false;

	m_nPreLightness  = m_nLightness  = DEFAULT_LIGHTNESS;         
	m_nPreSaturation = m_nSaturation = DEFAULT_SATURATION;      
	m_fSharpen       = DEFAULT_SHARPEN;                           
		m_nContrast = DEFAULT_CONTRAST;



    // Gets whether the device opens the Gamma enabled
    emStatus = __IsEnableDeviceGamma();
    VERIFY_STATUS_RET(emStatus);
	

	// Check whether the device supports Gamma parameters
	emStatus = GXIsImplemented(m_hDevice, GX_FLOAT_GAMMA_PARAM, &bIsImplemented);
	VERIFY_STATUS_RET(emStatus);
	if (bIsImplemented)
	{
		// Gamma parameters
		emStatus = GXGetFloat(m_hDevice, GX_FLOAT_GAMMA_PARAM, &m_dGamma);
		VERIFY_STATUS_RET(emStatus);
	}
	else
	{
		m_dGamma = DEFAULT_GAMMA;
	}
	m_dPreGamma = m_dGamma;
	
	// Determine if sharpening Settings are enabled
	emStatus = __IsEnableDeviceSharpness();
	VERIFY_STATUS_RET(emStatus);

	if (m_bColorFilter)
	{
		emStatus = __InitColorParam();
		VERIFY_STATUS_RET(emStatus);
	} 
	else
	{
		// If it's a black and white camera
		m_bEnableColorCorrect = false;
		m_bColorCorrection = false;
		m_nGetColorCorrection = DEFAULT_COLOR_CORRECTION;
		m_nPreColorCorrection = DEFAULT_COLOR_CORRECTION;
		m_nColorCorrection = DEFAULT_COLOR_CORRECTION;
	}
	
	return emStatus;
}

//---------------------------------------------------------------------------------
/**
\brief   Initializes some parameters of the color camera

\return  Initializes the resulting state value
*/
//----------------------------------------------------------------------------------
GX_STATUS CGxImageProcessDlg::__InitColorParam()
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	// Determine whether the camera is on color correction
	emStatus = __IsEnableDeviceColorCorrect();
	VERIFY_STATUS_RET(emStatus);


	// Determine whether the camera supports color correction parameter acquisition
	emStatus = GXIsImplemented(m_hDevice, GX_INT_COLOR_CORRECTION_PARAM, &m_bColorCorrection);
	VERIFY_STATUS_RET(emStatus);
	if (m_bColorCorrection)
	{
		// Gets the color correction parameter
		emStatus = GXGetInt(m_hDevice, GX_INT_COLOR_CORRECTION_PARAM, &m_nGetColorCorrection);
		VERIFY_STATUS_RET(emStatus);
	}
	else
	{
		m_nGetColorCorrection = DEFAULT_COLOR_CORRECTION;
	}

	// Check whether the user has selected the color correction
	if (m_bIsOpenCC)
	{
		m_nPreColorCorrection = m_nColorCorrection = m_nGetColorCorrection;
	}
	else
	{
		m_nPreColorCorrection = m_nColorCorrection = DEFAULT_COLOR_CORRECTION;
	}

	return GX_STATUS_SUCCESS;
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize image process struct

\return  true: Success, false: Set Fail
*/
//----------------------------------------------------------------------------------
bool CGxImageProcessDlg::__InitColorStruct()
{
	VxInt32 nStatus = DX_OK;

	do 
	{
		// Compute the array of image color adjustment 
		m_objImageProcess.nCCBufLength = CALCCCPARAM_LENGTH;
		m_objImageProcess.parrCC = new VxInt16[CALCCCPARAM_LENGTH];
		if (m_objImageProcess.parrCC == NULL)
		{
			nStatus = DX_NOT_ENOUGH_SYSTEM_MEMORY;
			break;
		}
		nStatus = DxCalcCCParam(m_nColorCorrection, m_nSaturation, m_objImageProcess.parrCC, CALCCCPARAM_LENGTH);
		if (nStatus != DX_OK)
		{
			break;
		}

		// Get the ImageProcess look up table
		nStatus = DxGetLut((VxInt32)m_nContrast, m_dGamma, m_nLightness, NULL, &m_objImageProcess.nLutLength);
		if (nStatus != DX_OK)
		{
			break;
		}
		m_objImageProcess.pProLut = new(std::nothrow) VxUint8[m_objImageProcess.nLutLength];
		if (m_objImageProcess.pProLut == NULL)
		{
			nStatus = DX_NOT_ENOUGH_SYSTEM_MEMORY;
			break;
		}
		nStatus = DxGetLut((VxInt32)m_nContrast, m_dGamma, m_nLightness, m_objImageProcess.pProLut, &m_objImageProcess.nLutLength);
		if (nStatus != DX_OK)
		{
			break;
		}

	} while (0);

	// Fail to initialize LUT,release resource 
	if (nStatus != DX_OK)
	{
		if (m_objImageProcess.parrCC != NULL)
		{
			delete[]m_objImageProcess.parrCC;
			m_objImageProcess.parrCC = NULL;
		}
		if (m_objImageProcess.pProLut != NULL)
		{
			delete[]m_objImageProcess.pProLut;
			m_objImageProcess.pProLut = NULL;
		}
		return false;
	}

	// Initialize other parameters for image processing            
	m_objImageProcess.bAccelerate            = m_bAccelerate;
	m_objImageProcess.bDefectivePixelCorrect = m_bDefectivePixelCorrect;
	m_objImageProcess.bDenoise               = m_bDenoise;
	m_objImageProcess.bFlip                  = true;
	m_objImageProcess.bSharpness             = m_bSharpness;
	m_objImageProcess.fSharpFactor           = m_fSharpen;
	m_objImageProcess.cvType                 = RAW2RGB_NEIGHBOUR;
	m_objImageProcess.emLayOut               = (DX_PIXEL_COLOR_FILTER)m_nPixelColorFilter;

	return true;
}

//---------------------------------------------------------------------------------
/**
\brief   Initializes the black and white image processing structure

\return  true:successfully set,false:Setup failed
*/
//----------------------------------------------------------------------------------
bool CGxImageProcessDlg::__InitMonoStruct()
{
	VxInt32 nStatus = DX_OK;

	do 
	{
		// Compute image processing Lut lookup tables
		nStatus = DxGetLut((VxInt32)m_nContrast, m_dGamma, m_nLightness, NULL, &m_objMonoImgProc.nLutLength);
		if (nStatus != DX_OK)
		{
			break;
		}
		m_objMonoImgProc.pProLut = new(std::nothrow) VxUint8[m_objMonoImgProc.nLutLength];
		if(m_objMonoImgProc.pProLut == NULL)
		{
			nStatus = DX_NOT_ENOUGH_SYSTEM_MEMORY;
			break;
		}

		nStatus = DxGetLut((VxInt32)m_nContrast, m_dGamma, m_nLightness, m_objMonoImgProc.pProLut, &m_objMonoImgProc.nLutLength);
	} while (0);

	// The image processing lookup table failed to initialize, freeing the resource
	if (nStatus != DX_OK)
	{
		if (m_objMonoImgProc.pProLut != NULL)
		{
			delete[]m_objMonoImgProc.pProLut;
			m_objMonoImgProc.pProLut = NULL;
		}
		return false;
	}

	// The black and white image processing function sets the structure to initialize other parameters
	m_objMonoImgProc.bAccelerate            = m_bAccelerate;
	m_objMonoImgProc.bDefectivePixelCorrect = m_bDefectivePixelCorrect;
	m_objMonoImgProc.bSharpness             = m_bSharpness;
	m_objMonoImgProc.fSharpFactor           = m_fSharpen;
	
	return true;
}

//---------------------------------------------------------------------------------
/**
\brief   Initializes the image processing structure

\return  true:successfully set ,false:Setup failed
*/
//----------------------------------------------------------------------------------
bool CGxImageProcessDlg::InitImageProcessStruct()
{
	if (m_bColorFilter)
	{
		return __InitColorStruct();
	} 
	else
	{
		return __InitMonoStruct();
	}
}

// ---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to set the pixel format to 8bit

\return  emStatus GX_STATUS_SUCCESS: Success,else: Fail
*/
// ----------------------------------------------------------------------------------
GX_STATUS CGxImageProcessDlg::SetPixelFormat8bit()
{
	uint32_t  i            = 0;
	int64_t   nPixelFormat = 0;                     
	uint32_t  nEnmuEntry   = 0;
	size_t    nBufferSize  = 0;	
	GX_STATUS emStatus     = GX_STATUS_SUCCESS;
	GX_ENUM_DESCRIPTION  *pEnumDescription = NULL;
	GX_ENUM_DESCRIPTION  *pEnumTemp        = NULL;
	
	// Get the feature PixelSize, this feature indicates the depth of the pixel values in the acquired images in bits per pixel
	emStatus = GXGetEnum(m_hDevice, GX_ENUM_PIXEL_FORMAT, &nPixelFormat);
	VERIFY_STATUS_RET(emStatus);
	
	// If the PixelSize is 8bit then return
	if ((nPixelFormat & GX_PIXEL_8BIT) == GX_PIXEL_8BIT)
	{
		return GX_STATUS_SUCCESS;
	}
	
	// Get the enumeration entry of the pixel format the device supports.
	emStatus = GXGetEnumEntryNums(m_hDevice, GX_ENUM_PIXEL_FORMAT, &nEnmuEntry);
	VERIFY_STATUS_RET(emStatus);
	
	// Allocate memory for getting the enumeration entry of the pixel format.
	nBufferSize      = nEnmuEntry * sizeof(GX_ENUM_DESCRIPTION);
	pEnumDescription = new GX_ENUM_DESCRIPTION[nEnmuEntry];
	
	// Gets the supported enumerated values
	emStatus = GXGetEnumDescription(m_hDevice, GX_ENUM_PIXEL_FORMAT, pEnumDescription, &nBufferSize);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		// release resource
		if (pEnumDescription != NULL)
		{
			delete []pEnumDescription;
			pEnumDescription = NULL;
		}
		return emStatus;
	}
	
	// A loop to visit every enumeration entry node once to find which enumeration entry node is 8bit and then set pixel format to its value 
	pEnumTemp = pEnumDescription;
	for (i = 0; i < nEnmuEntry; i++)
	{
		if ((pEnumTemp->nValue & GX_PIXEL_8BIT) == GX_PIXEL_8BIT)
		{
			emStatus = GXSetEnum(m_hDevice, GX_ENUM_PIXEL_FORMAT, pEnumTemp->nValue);
			break;
		}
		pEnumTemp++;
	}	
	
	// Release memory
	if (pEnumDescription != NULL)
	{
		delete []pEnumDescription;
		pEnumDescription = NULL;
	}
	
	return emStatus;
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize UI

\return  void 
*/
//----------------------------------------------------------------------------------
void CGxImageProcessDlg::InitUI()
{
	char chPutText[10] = {0};
	
	// Initialize BalanceWhite Combobox control 
    InitEnumUI(GX_ENUM_BALANCE_WHITE_AUTO, (CComboBox *)GetDlgItem(IDC_COMBO_AWB), m_bColorFilter);
	
	// Initialize AwbLampHouse Combobox control 
	InitEnumUI(GX_ENUM_AWB_LAMP_HOUSE, (CComboBox *)GetDlgItem(IDC_COMBO_AWB_LIGHT), m_bAwbLampHouse);
	
	// Initialize the range of slider
	m_sliderSharpen.SetRange(1, 50, TRUE);
	m_sliderGamma.SetRange(1, 100, TRUE);
	m_sliderContrast.SetRange(-50, 100, TRUE);
	m_sliderLightness.SetRange(-150, 150, TRUE);
	m_sliderSaturation.SetRange(0, 128, TRUE);
	
	// Initialize the position of slider
	m_sliderSharpen.SetPos(int(m_fSharpen * 10));
	m_sliderGamma.SetPos(int(m_dGamma * 10));
	m_sliderContrast.SetPos(int(m_nContrast));	
	m_sliderLightness.SetPos(m_nLightness);
	m_sliderSaturation.SetPos(m_nSaturation);

	// Initialize the value of edit control
	_snprintf(chPutText, sizeof(chPutText), "%f", m_fSharpen);
	GetDlgItem(IDC_EDIT_SHARPEN)->SetWindowText(chPutText);
	_snprintf(chPutText, sizeof(chPutText), "%f", m_dGamma);
	GetDlgItem(IDC_EDIT_GAMMA)->SetWindowText(chPutText);
	_snprintf(chPutText, sizeof(chPutText), "%lld", m_nContrast);
	GetDlgItem(IDC_EDIT_CONTRAST)->SetWindowText(chPutText);
	_snprintf(chPutText, sizeof(chPutText), "%d", m_nLightness);
	GetDlgItem(IDC_EDIT_LIGHTNESS)->SetWindowText(chPutText);
	_snprintf(chPutText, sizeof(chPutText), "%d",m_nSaturation);
	GetDlgItem(IDC_EDIT_SATURATION)->SetWindowText(chPutText);
	
	// Initialize the new check box
	if (m_bColorCorrection)
	{
		((CButton*)GetDlgItem(IDC_OPEN_COLOR_CORRECT))->SetCheck(m_bIsOpenCC && !m_bEnableColorCorrect);
	}
	else
	{
		((CButton*)GetDlgItem(IDC_OPEN_COLOR_CORRECT))->SetCheck(!m_bIsOpenCC && !m_bEnableColorCorrect);
	}

	((CButton*)GetDlgItem(IDC_OPEN_DENOISE))->SetCheck(m_bDenoise);
	((CButton*)GetDlgItem(IDC_OPEN_SHARPEN))->SetCheck(m_bSharpness);
	((CButton*)GetDlgItem(IDC_OPEN_ACCELERATE))->SetCheck(m_bAccelerate);
	((CButton*)GetDlgItem(IDC_OPEN_BAD_PIXEL_CORRECT))->SetCheck(m_bDefectivePixelCorrect);

}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize the UI of ComboBox 
\param   emFeatureID   [in]    
\param   pComboBox     [in]    
\param   bIsImplement  [in]    

\return  void
*/
//----------------------------------------------------------------------------------
void CGxImageProcessDlg::InitEnumUI(GX_FEATURE_ID_CMD emFeatureID, CComboBox *pComboBox, bool bIsImplement)
{
	// Check whether the ComboBox is available
	if ((pComboBox == NULL) || (!bIsImplement))
	{
		return;
	}
	
	GX_ENUM_DESCRIPTION *pEnum = NULL;
	GX_STATUS emStatus    = GX_STATUS_ERROR;
	size_t    nbufferSize = 0;
	uint32_t  nEntryNum   = 0;
	int64_t   nEnumValue  = -1;
	int       nCursel     = 0;
	
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
	
	// Gets an enumeration description of the functionality
	emStatus = GXGetEnumDescription(m_hDevice, emFeatureID, pEnum, &nbufferSize);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		if (pEnum != NULL)
		{  
			delete []pEnum;
			pEnum = NULL;
		}
		ShowErrorString(emStatus);
		return;
	}
		
	// Get the enumeration value 
	emStatus = GXGetEnum(m_hDevice, emFeatureID, &nEnumValue);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		if (pEnum != NULL)
		{  
			delete []pEnum;
			pEnum = NULL;
		}
		ShowErrorString(emStatus);
		return;
	}

	// If BalanceWhiteAuto has been selected then update the value
	if (emFeatureID == GX_ENUM_BALANCE_WHITE_AUTO)
	{
		m_nAutoWhiteBalance = nEnumValue;
	}

	// Initialize the options of the ComboBox
	pComboBox->ResetContent();
	for (uint32_t i = 0; i < nEntryNum; i++)
	{
		pComboBox->SetItemData(pComboBox->AddString(pEnum[i].szSymbolic), (uint32_t)pEnum[i].nValue);
		if (pEnum[i].nValue == nEnumValue)
		{
			nCursel = i;
		}
	}
	
	// Show the value to the UI
	pComboBox->SetCurSel(nCursel);
	
	// Release memory  
	if (pEnum != NULL)
	{  
		delete []pEnum;
		pEnum = NULL;
	}
}

//This function demonstrates how to start acquisition
void CGxImageProcessDlg::OnStartSnap() 
{
	bool      bReturn  = false;
    GX_STATUS emStatus = GX_STATUS_SUCCESS;
	
	// Allocate memory for showing image 
	bReturn = PrepareForShowImage();
	if (!bReturn)
	{
        MessageBox("Fail to allocate resource for showing image!");
		return;
	}
	
	// Register the CaptureCallback function 
	emStatus = GXRegisterCaptureCallback(m_hDevice, this, OnFrameCallbackFun);

	if (emStatus != GX_STATUS_SUCCESS)
	{
		UnPrepareForShowImage();
		ShowErrorString(emStatus);
		return;
	}

	//Set StreamBufferHandlingMode
	emStatus = GXSetEnum(m_hDevice, GX_DS_ENUM_STREAM_BUFFER_HANDLING_MODE, GX_DS_STREAM_BUFFER_HANDLING_MODE_OLDEST_FIRST);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		UnPrepareForShowImage();
		ShowErrorString(emStatus);
		return;
	}
	
	// Send AquisitionStart commands
	emStatus = GXSendCommand(m_hDevice, GX_COMMAND_ACQUISITION_START);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		UnPrepareForShowImage();
		ShowErrorString(emStatus);
		return;
	}
	m_bIsSnap = true;

	UpdateUI();
}

//----------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to allocate memory for showing image

\return  true: Success, false: Fail
*/
//----------------------------------------------------------------------------------
bool CGxImageProcessDlg::PrepareForShowImage()
{
	// Initialize bitmap header
	m_pBmpInfo								= (BITMAPINFO *)m_chBmpBuf;
	m_pBmpInfo->bmiHeader.biSize			= sizeof(BITMAPINFOHEADER);
	m_pBmpInfo->bmiHeader.biWidth			= (LONG)m_nImageWidth;
	m_pBmpInfo->bmiHeader.biHeight			= (LONG)m_nImageHeight;	
	m_pBmpInfo->bmiHeader.biPlanes			= 1;
	int nChannel                            = 0; //Color three channel, black and white single channel

	do
	{
		// Color/black and white camera
	    if (m_bColorFilter)
	    {
	    	m_pBmpInfo->bmiHeader.biBitCount	= 24;
	    	nChannel = 3;
	    } 
	    else
	    {
		    m_pBmpInfo->bmiHeader.biBitCount	= 8;
		    nChannel = 1;
		
		    // Black and white images need to be initialized to the palette
		    for(int i=0;i < 256;i++)
		    {
		    	m_pBmpInfo->bmiColors[i].rgbBlue	=i;
		    	m_pBmpInfo->bmiColors[i].rgbGreen	=i;
		    	m_pBmpInfo->bmiColors[i].rgbRed		=i;
		    	m_pBmpInfo->bmiColors[i].rgbReserved=i;
		    }
    
		    m_pBufferMono = new BYTE[(size_t)(m_nImageWidth * m_nImageHeight)];
	        if (m_pBufferMono == NULL)
	        {
		        break;
	        }
	    }
	
	    m_pBmpInfo->bmiHeader.biCompression		= BI_RGB;
	    m_pBmpInfo->bmiHeader.biSizeImage		= 0;
	    m_pBmpInfo->bmiHeader.biXPelsPerMeter	= 0;
	    m_pBmpInfo->bmiHeader.biYPelsPerMeter	= 0;
	    m_pBmpInfo->bmiHeader.biClrUsed			= 0;
	    m_pBmpInfo->bmiHeader.biClrImportant	= 0;
	
	    // Request space for storing RGB/Mono image data
	    m_pBufferImage = new BYTE[(size_t)(m_nImageWidth * m_nImageHeight * nChannel)];
	    if (m_pBufferImage == NULL)
	    {
	    	 break;
	    }

	    // Allocate memory for showing raw image  
	    m_pBufferRaw = new BYTE[(size_t)m_nPayLoadSize];
	    if (m_pBufferRaw == NULL)
	    {
		     break;
	    }

	return true;
	}while(0);

	if(m_pBufferMono != NULL)
	{
		delete []m_pBufferMono;
	    m_pBufferMono = NULL;
	}

	if (m_pBufferImage != NULL)
	{
		delete []m_pBufferImage;
	    m_pBufferImage = NULL;
	}

	if (m_pBufferRaw != NULL)
	{
		delete []m_pBufferImage;
	    m_pBufferImage = NULL;
	}

	return false;
	
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to release memory for showing image 

\return  void
*/
//----------------------------------------------------------------------------------
void CGxImageProcessDlg::UnPrepareForShowImage()
{
	// If the program is in the process of image processing, wait for the processing to complete before releasing the resources
	while(m_bImageProcess == true)
	{
		Sleep(200);
	}

	if (m_pBufferImage != NULL)
	{
		delete []m_pBufferImage;
		m_pBufferImage = NULL;
	}
	if (m_pBufferRaw != NULL)
	{
		delete []m_pBufferRaw;
		m_pBufferRaw = NULL;
	}
	if (m_pBufferMono != NULL)
	{
		delete []m_pBufferMono;
		m_pBufferMono = NULL;
	}
	
}

//---------------------------------------------------------------------------------
/**
\brief   This is a acquisition callback function that demonstrates how to process the acquired image
\param   pFrame   

\return  void
*/
//----------------------------------------------------------------------------------
void CGxImageProcessDlg::OnColorImageProcess()
{
	VxInt32  emStatus = DX_OK;

	m_objImageProcess.bAccelerate = m_bAccelerate;
	m_objImageProcess.bDefectivePixelCorrect = m_bDefectivePixelCorrect;
	m_objImageProcess.bSharpness = m_bSharpness;
	m_objImageProcess.bDenoise = m_bDenoise;

	// Determines whether to recalculate the image processing color adjustment array
	if ((m_nPreSaturation != m_nSaturation) || (m_nColorCorrection != m_nPreColorCorrection))
	{
		DxCalcCCParam(m_nColorCorrection, m_nSaturation, m_objImageProcess.parrCC, CALCCCPARAM_LENGTH);
		m_nPreSaturation      = m_nSaturation;
		m_nPreColorCorrection = m_nColorCorrection;
	}

	// Determine whether the image processing 8-bit lookup table needs to be recalculated
	if ((m_nPreContrast != m_nContrast) || (m_nPreLightness != m_nLightness) 
		|| (m_dPreGamma != m_dGamma))
	{
		DxGetLut((VxInt32)m_nContrast, m_dGamma, m_nLightness, m_objImageProcess.pProLut, &m_objImageProcess.nLutLength);
		m_nPreContrast  = m_nContrast;
		m_nPreLightness = m_nLightness;
		m_dPreGamma     = m_dGamma;
	}

	// Raw8 image is processed
	emStatus = DxRaw8ImgProcess(m_pBufferRaw, m_pBufferImage, m_nImageWidth, m_nImageHeight, &m_objImageProcess);
	if (emStatus == DX_CPU_NOT_SUPPORT_ACCELERATE)
	{
		if (m_bUpData)
		{
			GetDlgItem(IDC_STATIC_CPU_NOT_SUPPORT)->SetWindowText("(SSSE3 instruction set acceleration CPU is not supported!)");
			m_bUpData = false;	
		}
		return;
	}

	// display image
	DrawImage(m_pBufferImage, m_nImageWidth, m_nImageHeight, m_pBmpInfo);
}

//---------------------------------------------------------------------------------
/**
\brief   Callback function in the black and white image algorithm processing

\return  
*/
//----------------------------------------------------------------------------------
void CGxImageProcessDlg::OnMonoImageProcess()
{
	VxInt32  emStatus = DX_OK;
	m_objMonoImgProc.bAccelerate = m_bAccelerate;
	m_objMonoImgProc.bDefectivePixelCorrect = m_bDefectivePixelCorrect;
	m_objMonoImgProc.bSharpness = m_bSharpness;
	m_objMonoImgProc.fSharpFactor = m_fSharpen;
	

	// Check whether the 8bit LUT needs to be recomputed 
	if ((m_nPreContrast != m_nContrast) || (m_nPreLightness != m_nLightness) || (m_dPreGamma != m_dGamma))
	{
		DxGetLut((VxInt32)m_nContrast, m_dGamma, m_nLightness, m_objMonoImgProc.pProLut, &m_objMonoImgProc.nLutLength);
		m_nPreContrast  = m_nContrast;
		m_nPreLightness = m_nLightness;
		m_dPreGamma     = m_dGamma;
	}

	// Raw8 image is processed
	emStatus = DxMono8ImgProcess(m_pBufferRaw, m_pBufferMono, m_nImageWidth, m_nImageHeight, &m_objMonoImgProc);
	if (emStatus == DX_CPU_NOT_SUPPORT_ACCELERATE)
	{
		if (m_bUpData)
		{
			GetDlgItem(IDC_STATIC_CPU_NOT_SUPPORT)->SetWindowText("(SSSE3 instruction set acceleration CPU is not supported!)");
			m_bUpData = false;	
		}
		return;
	}

	// The black and white camera needs to flip the data before displaying
	for(int i =0;i <m_nImageHeight;i++)
	{
		memcpy(m_pBufferImage + i * m_nImageWidth, m_pBufferMono + (m_nImageHeight - i -1) * m_nImageWidth,(size_t)m_nImageWidth);
	}

	DrawImage(m_pBufferImage, m_nImageWidth, m_nImageHeight, m_pBmpInfo);
}


//---------------------------------------------------------------------------------
/**
\brief   callback function
\param   pFrame   The callback parameter

\return  
*/
//----------------------------------------------------------------------------------
void __stdcall CGxImageProcessDlg::OnFrameCallbackFun(GX_FRAME_CALLBACK_PARAM* pFrame)
{
	CGxImageProcessDlg *pDlg = (CGxImageProcessDlg*)(pFrame->pUserParam);
	
	// The return state of the current frame image
	if (pFrame->status != 0)
	{
		return;
	}
	
	// The identification program is performing image processing
	pDlg->m_bImageProcess = true;

	// Copies the image data from the callback
	memcpy(pDlg->m_pBufferRaw, pFrame->pImgBuf, pFrame->nImgSize);

	if (pDlg->m_bColorFilter)
	{
		pDlg->OnColorImageProcess();
	} 
	else
	{
		pDlg->OnMonoImageProcess();
	}
	
	// The identifier does not perform image processing
	pDlg->m_bImageProcess = false;
}

//---------------------------------------------------------------------------------
/**
\brief   Display image
\param   pImageBuf  
\param   nWidth     
\param   nHeight     
\param   pBmpInfo    
\param   pWnd       
\return  void
*/
//----------------------------------------------------------------------------------
void CGxImageProcessDlg::DrawImage(BYTE *pImageBuf, int nWidth, int nHeight, BITMAPINFO *pBmpInfo)
{
	if ((pImageBuf == NULL) || (pBmpInfo == NULL))
	{
		return;
	}

	// Displays the window size of the image                                      
	int nWndWidth  = m_objRect.right - m_objRect.left;     
	int nWndHeight = m_objRect.bottom - m_objRect.top;      

	// You could use the following code to show image. 
	::SetStretchBltMode(m_hDC, COLORONCOLOR);
	::StretchDIBits(m_hDC,
		0,						
		0,
		nWndWidth,
		nWndHeight,
		0,
		0,
		nWidth,
		nHeight,
		pImageBuf,
		pBmpInfo,
		DIB_RGB_COLORS,
		SRCCOPY);		
}

//This function demonstrates how to stop acquisition
void CGxImageProcessDlg::OnStopSnap() 
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	
	// Send AcquisitionStop command
	emStatus = GXSendCommand(m_hDevice, GX_COMMAND_ACQUISITION_STOP);
	GX_VERIFY(emStatus);
	
	// Unregister the callback function
	emStatus = GXUnregisterCaptureCallback(m_hDevice);
	GX_VERIFY(emStatus);
    
	// Release memory for showing image  
	UnPrepareForShowImage();
	m_bIsSnap = false;
	
	UpdateUI();	
}

void CGxImageProcessDlg::OnCloseDevice() 
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	
	// If the device is acquiring image then stop it.
	if (m_bIsSnap)
	{
		// Send stop collection command
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
		
		// Release memory for showing image
		UnPrepareForShowImage();
		m_bIsSnap = false;
	}
	
	// Close the camera
	emStatus = GXCloseDevice(m_hDevice);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		// TODO: Add your error handler code here 
	}
	m_hDevice = NULL;

	// Release memory for image process
	UnInitImageProcessStruct();
	m_bIsOpen = false;
	m_bEnableOpen = true;

	// StopTimer
	KillTimer(0);
	
	// Update UI interface
	UpdateUI();
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to release memory for image process

\return  void
*/
//----------------------------------------------------------------------------------
void CGxImageProcessDlg::UnInitImageProcessStruct()
{
	if (m_objImageProcess.parrCC != NULL)
	{
		delete[]m_objImageProcess.parrCC;
		m_objImageProcess.parrCC = NULL;
	}
	if (m_objImageProcess.pProLut != NULL)
	{
		delete[]m_objImageProcess.pProLut;
		m_objImageProcess.pProLut = NULL;
	}
	if (m_objMonoImgProc.pProLut != NULL)
	{
		delete[]m_objMonoImgProc.pProLut;
		m_objMonoImgProc.pProLut = NULL;
	}
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to convert the error code to error description
\param  emErrorStatus  [in]   

\return void
*/
//----------------------------------------------------------------------------------
void CGxImageProcessDlg::ShowErrorString(GX_STATUS emErrorStatus)
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
	
	// Display error description information
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
\brief   This function demonstrates how to update UI

\return  void
*/
//----------------------------------------------------------------------------------
void CGxImageProcessDlg::UpdateUI()
{
	GetDlgItem(IDC_OPEN_DEVICE)->EnableWindow(m_bEnableOpen);
	GetDlgItem(IDC_START_SNAP)->EnableWindow(m_bIsOpen && (!m_bIsSnap));
	GetDlgItem(IDC_STOP_SNAP)->EnableWindow(m_bIsOpen && m_bIsSnap);
	GetDlgItem(IDC_CLOSE_DEVICE)->EnableWindow(m_bIsOpen);

	GetDlgItem(IDC_COMBO_AWB)->EnableWindow(m_bIsOpen && m_bColorFilter);
	GetDlgItem(IDC_COMBO_AWB_LIGHT)->EnableWindow(m_bIsOpen && m_bAwbLampHouse && m_bColorFilter);
	GetDlgItem(IDC_OPEN_ACCELERATE)->EnableWindow(m_bIsOpen);
	GetDlgItem(IDC_OPEN_BAD_PIXEL_CORRECT)->EnableWindow(m_bIsOpen);
	GetDlgItem(IDC_OPEN_DENOISE)->EnableWindow(m_bIsOpen && m_bColorFilter);
	GetDlgItem(IDC_OPEN_SHARPEN)->EnableWindow(m_bIsOpen && !m_bEnableSharpness);
	GetDlgItem(IDC_OPEN_COLOR_CORRECT)->EnableWindow(m_bIsOpen && m_bColorCorrection && !m_bEnableColorCorrect && m_bColorFilter);
	GetDlgItem(IDC_SLIDER_SHARPEN)->EnableWindow(m_bIsOpen && !m_bEnableSharpness && m_bSharpness);
	GetDlgItem(IDC_SLIDER_GAMMA)->EnableWindow(m_bIsOpen && !m_bEnableGamma);
	GetDlgItem(IDC_SLIDER_CONTRAST)->EnableWindow(m_bIsOpen);
	GetDlgItem(IDC_SLIDER_LIGHTNESS)->EnableWindow(m_bIsOpen);
	GetDlgItem(IDC_SLIDER_SATURATION)->EnableWindow(m_bIsOpen && m_bColorFilter);
	
	GetDlgItem(IDC_COMBO_ChooseDevice)->EnableWindow(!m_bIsOpen);
	GetDlgItem(IDC_EDIT_GAMMA)->EnableWindow(m_bIsOpen && !m_bEnableGamma);
	GetDlgItem(IDC_EDIT_SATURATION)->EnableWindow(m_bIsOpen && m_bColorFilter);
	GetDlgItem(IDC_EDIT_SHARPEN)->EnableWindow(m_bIsOpen && !m_bEnableSharpness && m_bSharpness);
}

void CGxImageProcessDlg::OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar) 
{
	char PutText[10] = {0};
	int  nValue      = 0;

	if (pScrollBar == (CScrollBar*)&m_sliderLightness)            // The Lightness slider
	{
		m_nLightness = m_sliderLightness.GetPos();
		_snprintf(PutText, sizeof(PutText), "%d", m_nLightness);
		GetDlgItem(IDC_EDIT_LIGHTNESS)->SetWindowText(PutText);	
	}
	else if (pScrollBar == (CScrollBar*)&m_sliderContrast)         // The Contrast slider
	{
		m_nContrast = m_sliderContrast.GetPos();
		_snprintf(PutText, sizeof(PutText), "%lld", m_nContrast);
		GetDlgItem(IDC_EDIT_CONTRAST)->SetWindowText(PutText);	
	}
	else if (pScrollBar == (CScrollBar*)&m_sliderSaturation)       // The Saturation slider
	{
		m_nSaturation = m_sliderSaturation.GetPos();
		_snprintf(PutText, sizeof(PutText), "%d", m_nSaturation);
		GetDlgItem(IDC_EDIT_SATURATION)->SetWindowText(PutText);	
	}	
	else if (pScrollBar == (CScrollBar*)&m_sliderSharpen)          // The Sharpen slider
	{
		nValue = m_sliderSharpen.GetPos();
		m_fSharpen = nValue / 10.0F;
		m_objImageProcess.fSharpFactor = m_fSharpen;
		_snprintf(PutText, sizeof(PutText), "%7f", m_fSharpen);
		GetDlgItem(IDC_EDIT_SHARPEN)->SetWindowText(PutText);		
	}
	else if (pScrollBar == (CScrollBar*)&m_sliderGamma)            // The Gamma slider
	{
		nValue = m_sliderGamma.GetPos();
		m_dGamma = nValue / 10.0;
		_snprintf(PutText, sizeof(PutText), "%7f", m_dGamma);
		GetDlgItem(IDC_EDIT_GAMMA)->SetWindowText(PutText);		
	}
	UpdateData(FALSE);
	
	CDialog::OnHScroll(nSBCode, nPos, pScrollBar);
}

void CGxImageProcessDlg::OnOpenBadPixelCorrect() 
{
	m_bDefectivePixelCorrect = !m_bDefectivePixelCorrect;
}

void CGxImageProcessDlg::OnOpenDenoise() 
{
	m_bDenoise = !m_bDenoise;
}

void CGxImageProcessDlg::OnOpenAccelerate() 
{
	m_bAccelerate = !m_bAccelerate;
}

void CGxImageProcessDlg::OnOpenSharpen() 
{
	m_bSharpness = !m_bSharpness;

	UpdateUI();
}

void CGxImageProcessDlg::OnOpenColorCorrect() 
{
	m_bIsOpenCC = !m_bIsOpenCC;
	if (m_bIsOpenCC)
	{
		m_nColorCorrection = m_nGetColorCorrection;
	}
	else
	{
		m_nColorCorrection = DEFAULT_COLOR_CORRECTION;
	}	
}

void CGxImageProcessDlg::OnSelchangeComboAwb() 
{
	CComboBox *pBox    = (CComboBox *)GetDlgItem(IDC_COMBO_AWB);
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	int64_t   nEnumVal = 0;
	int       nIndex   = 0;
	
	nIndex   = pBox->GetCurSel();                    // Get current selection
	nEnumVal = (int64_t)pBox->GetItemData(nIndex);   // Get the enumeration entry of current selection
	
	// Set BalanceWhite
	emStatus = GXSetEnum(m_hDevice, GX_ENUM_BALANCE_WHITE_AUTO, nEnumVal);
	GX_VERIFY(emStatus);

	m_nAutoWhiteBalance = nEnumVal;
}

void CGxImageProcessDlg::OnSelchangeComboAwbLight() 
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	CComboBox *pBox    = (CComboBox *)GetDlgItem(IDC_COMBO_AWB_LIGHT);
	
	int      nIndex    = pBox->GetCurSel();                   // Get current selection
	int64_t  nEnumVal  = (int64_t)pBox->GetItemData(nIndex);  // Get the enumeration entry of current selection
	
	// Set AwbLampHouse
	emStatus = GXSetEnum(m_hDevice, GX_ENUM_AWB_LAMP_HOUSE, nEnumVal);
	GX_VERIFY(emStatus);
}


BOOL CGxImageProcessDlg::PreTranslateMessage(MSG* pMsg) 
{
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

//----------------------------------------------------------------
/**
\brief  This is a callback function of timer for updating UI.
\param  
 
\return void
*/
//----------------------------------------------------------------
void CGxImageProcessDlg::OnTimer(UINT_PTR nIDEvent)
{
	GX_STATUS status = GX_STATUS_ERROR;

	//Get the current BalanceWhiteAuto value from the device and then update to UI
	if (m_nAutoWhiteBalance == GX_BALANCE_WHITE_AUTO_ONCE)
	{
		int64_t nWBEnumVal = 0;   //< Holds the enumeration value of the white balance

		CComboBox *pAutoWBBox = (CComboBox *)GetDlgItem(IDC_COMBO_AWB);

		//Get the enumeration entry of BalanceWhiteAuto
		status = GXGetEnum(m_hDevice, GX_ENUM_BALANCE_WHITE_AUTO, &nWBEnumVal);
		if (status != GX_STATUS_SUCCESS)
		{
			return;
		}

		m_nAutoWhiteBalance = nWBEnumVal;

		//Check whether the BalanceWhiteAuto is OFF
		if (m_nAutoWhiteBalance == GX_BALANCE_WHITE_AUTO_OFF)
		{
			for (int i = 0; i < WHITE_BALANCE_RNTRY; i++)
			{
				if ((int64_t)pAutoWBBox->GetItemData(i) == GX_BALANCE_WHITE_AUTO_OFF)
				{
					// Select the OFF item in the auto exposure control, which changes from ONCE to OFF
					pAutoWBBox->SetCurSel(i);
					break;
				}
			}
		}
	}

	CDialog::OnTimer(nIDEvent);
}

//----------------------------------------------------------------
/**
\brief  Gets whether the device opens color correction

\return GX_STATUS
*/
//----------------------------------------------------------------
GX_STATUS CGxImageProcessDlg::__IsEnableDeviceColorCorrect()
{
    GX_STATUS emStatus = GX_STATUS_SUCCESS;
    bool bIsImplement = false;
    m_bEnableColorCorrect = false;

    do 
    {
        emStatus = GXIsImplemented(m_hDevice, GX_BOOL_COLOR_TRANSFORMATION_ENABLE, &bIsImplement);
        VERIFY_STATUS_BREAK(emStatus);

        if(false == bIsImplement)
        {
            break;
        }

        emStatus = GXGetBool(m_hDevice, GX_BOOL_COLOR_TRANSFORMATION_ENABLE, &m_bEnableColorCorrect);
        VERIFY_STATUS_BREAK(emStatus);
    } while (0);
    
    return emStatus;
}

//----------------------------------------------------------------
/**
\brief  Gets whether the device opens the Gamma enabled

\return GX_STATUS
*/
//----------------------------------------------------------------
GX_STATUS CGxImageProcessDlg::__IsEnableDeviceGamma()
{
    GX_STATUS emStatus = GX_STATUS_SUCCESS;
    bool bIsImplement = false;
    m_bEnableGamma = false;

    do 
    {
        emStatus = GXIsImplemented(m_hDevice, GX_BOOL_GAMMA_ENABLE, &bIsImplement);
        VERIFY_STATUS_BREAK(emStatus);

        if(false == bIsImplement)
        {
            break;
        }

        emStatus = GXGetBool(m_hDevice, GX_BOOL_GAMMA_ENABLE, &m_bEnableGamma);
        VERIFY_STATUS_BREAK(emStatus);
    } while (0);

    return emStatus;
}

//----------------------------------------------------------------
/**
\brief  Gets whether the device opens sharpening enable

\return GX_STATUS
*/
//----------------------------------------------------------------
GX_STATUS CGxImageProcessDlg::__IsEnableDeviceSharpness()
{
    GX_STATUS emStatus = GX_STATUS_SUCCESS;
    bool bIsImplement = false;
    int64_t nValue = 0;
    m_bEnableSharpness = false;

    do 
    {
        emStatus = GXIsImplemented(m_hDevice, GX_ENUM_SHARPNESS_MODE, &bIsImplement);
        VERIFY_STATUS_BREAK(emStatus);

        if(false == bIsImplement)
        {
            break;
        }

        emStatus = GXGetEnum(m_hDevice, GX_ENUM_SHARPNESS_MODE, &nValue);
        VERIFY_STATUS_BREAK(emStatus);

        if(GX_SHARPNESS_MODE_OFF == nValue)
        {
            m_bEnableSharpness = false;
        }else if(GX_SHARPNESS_MODE_ON)
        {
            m_bEnableSharpness = true;
        }
    } while (0);

    return emStatus;
}

