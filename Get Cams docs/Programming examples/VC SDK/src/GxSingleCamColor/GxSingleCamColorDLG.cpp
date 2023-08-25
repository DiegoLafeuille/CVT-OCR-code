// GxSingleCamColorDlg.cpp : implementation file
//

#include "stdafx.h"
#include "GxSingleCamColor.h"
#include "GxSingleCamColorDlg.h"


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
// CGxSingleCamColorDlg dialog

CGxSingleCamColorDlg::CGxSingleCamColorDlg(CWnd* pParent /*=NULL*/)
	: CDialog(CGxSingleCamColorDlg::IDD, pParent)
	, m_bDevOpened(FALSE)
	, m_bIsSnap(FALSE)
	, m_bIsSaveImg(FALSE)
	, m_bIsImproveImg(FALSE)
	, m_bTriggerMode(false)
	, m_bTriggerActive(false)
	, m_bTriggerSource(false)
	, m_bColorFilter(false)
	, m_bBalanceRatioSelector(false)
	, m_nTriggerMode(GX_TRIGGER_MODE_OFF)
	, m_hDevice(NULL)
	, m_pBmpInfo(NULL)
	, m_pBufferRaw(NULL)
	, m_pBufferRGB(NULL)
	, m_pGammaLut(NULL)
	, m_pContrastLut(NULL)
	, m_nPayLoadSize(0)
	, m_nImageHeight(0)
	, m_nImageWidth(0)
	, m_nPixelColorFilter(GX_COLOR_FILTER_NONE)
	, m_nContrast(0)
	, m_nLutLength(0)
	, m_nColorCorrection(0)
	, m_nRatioSelector(-1)
	, m_dGamma(0.0)
	, m_nAutoWhiteBalance(0)
	, m_strFilePath("")
	, m_pWnd(NULL)
	, m_hDC(NULL)
{
	//{{AFX_DATA_INIT(CGxSingleCamColorDlg)
	m_dBalanceRatio = 0.0;
	m_dGainValue = 0.0;
	m_dShutterValue = 0.0;
	//}}AFX_DATA_INIT
	// Note that LoadIcon does not require a subsequent DestroyIcon in Win32
	m_hIcon    = AfxGetApp()->LoadIcon(IDR_MAINFRAME);

	memset(&m_stGainFloatRange,0,sizeof(GX_FLOAT_RANGE));
	memset(&m_stShutterFloatRange,0,sizeof(GX_FLOAT_RANGE));
	memset(&m_stFloatRange,0,sizeof(GX_FLOAT_RANGE));
	memset(m_chBmpBuf,0,sizeof(m_chBmpBuf));
}

void CGxSingleCamColorDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CGxSingleCamColorDlg)
	DDX_Text(pDX, IDC_EDIT_BALANCE_RATIO, m_dBalanceRatio);
	DDX_Text(pDX, IDC_EDIT_GAIN, m_dGainValue);
	DDX_Text(pDX, IDC_EDIT_SHUTTER, m_dShutterValue);
	//}}AFX_DATA_MAP
}

BEGIN_MESSAGE_MAP(CGxSingleCamColorDlg, CDialog)
	//{{AFX_MSG_MAP(CGxSingleCamColorDlg)
	ON_WM_QUERYDRAGICON()
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_CLOSE()
	ON_BN_CLICKED(IDC_BTN_SOFTTRIGGER, OnBnClickedBtnSofttrigger)
	ON_CBN_SELCHANGE(IDC_COMBO_TRIGGER_MODE, OnCbnSelchangeComboTriggerMode)
	ON_CBN_SELCHANGE(IDC_COMBO_AWB, OnCbnSelchangeComboWriteBlance)
    ON_CBN_SELCHANGE(IDC_COMBO_TRIGGER_SOURCE, OnCbnSelchangeComboTriggerSource)
    ON_CBN_SELCHANGE(IDC_COMBO_TRIGGER_ACTIVE, OnCbnSelchangeComboTriggerActive)
    ON_BN_CLICKED(IDC_BTN_OPEN_DEVICE, OnBnClickedBtnOpenDevice)
    ON_BN_CLICKED(IDC_BTN_CLOSE_DEVICE, OnBnClickedBtnCloseDevice)
    ON_BN_CLICKED(IDC_BTN_STOP_SNAP, OnBnClickedBtnStopSnap)
    ON_BN_CLICKED(IDC_BTN_START_SNAP, OnBnClickedBtnStartSnap)
    ON_BN_CLICKED(IDC_CHECK_SAVE_BMP, OnBnClickedCheckSaveBmp)
	ON_BN_CLICKED(IDC_CHECK_IMAGE_IMPROVE, OnBnClickedCheckImageImprove)
    ON_CBN_SELCHANGE(IDC_COMBO_RATIO_SELECTOR, OnCbnSelchangeComboRatioSelector)
	ON_EN_KILLFOCUS(IDC_EDIT_BALANCE_RATIO, OnEnKillfocusEditBalanceRatio)
	ON_EN_KILLFOCUS(IDC_EDIT_SHUTTER, OnKillfocusEditShutter)
	ON_EN_KILLFOCUS(IDC_EDIT_GAIN, OnKillfocusEditGain)
	ON_WM_TIMER()
	//}}AFX_MSG_MAP
	
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CGxSingleCamColorDlg message handlers

BOOL CGxSingleCamColorDlg::OnInitDialog()
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

	// Set the path for saving image 
	CString strDate = "";
	char    szName[MAX_PATH] = {0};
	CString strAppFullName = ""; 
	GetModuleFileName(NULL, szName, MAX_PATH);
	strAppFullName = szName;
	int nPos = strAppFullName.ReverseFind('\\');
	m_strFilePath += strAppFullName.Left(nPos);
	m_strFilePath += "\\GxSingleCamColorImages";

	//Get the pointer and DC handle for showing image
	m_pWnd = GetDlgItem(IDC_STATIC_SHOW_FRAME);
	m_hDC  = ::GetDC(m_pWnd->m_hWnd);

	UpDateUI();
	return TRUE;  // return TRUE  unless you set the focus to a control
}

void CGxSingleCamColorDlg::OnSysCommand(UINT nID, LPARAM lParam)
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

void CGxSingleCamColorDlg::OnPaint() 
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
HCURSOR CGxSingleCamColorDlg::OnQueryDragIcon()
{
	return (HCURSOR) m_hIcon;
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to close the application.

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamColorDlg::OnClose() 
{
	// TODO: Add your message handler code here and/or call default
	KillTimer(0);

	GX_STATUS emStatus = GX_STATUS_SUCCESS;

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

		if (m_pGammaLut != NULL)
		{
			delete[] m_pGammaLut;
			m_pGammaLut = NULL;
		}
		if (m_pContrastLut != NULL)
		{
			delete[] m_pContrastLut;
			m_pContrastLut = NULL;
		}	
	}

	// Release GxIAPI
	emStatus = GXCloseLib();

	CDialog::OnClose();
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to get the parameters to improve the quality of image 

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamColorDlg::GetImageImproveParam()
{
	if (!m_bColorFilter)
	{
		return;
	}

	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	VxInt32   nStatus  = DX_OK;
	bool      bIsImplemented = false;

	// Check whether the device supports contrast parameters
	emStatus = GXIsImplemented(m_hDevice, GX_INT_CONTRAST_PARAM, &bIsImplemented);
	GX_VERIFY(emStatus);
	if (bIsImplemented)
	{
		emStatus = GXGetInt(m_hDevice, GX_INT_CONTRAST_PARAM, &m_nContrast);
		GX_VERIFY(emStatus);
	}
	else
	{
		m_nContrast = DEFAULT_CONTRAST;
	}

	// Check whether the device supports Gamma parameters
	emStatus = GXIsImplemented(m_hDevice, GX_FLOAT_GAMMA_PARAM, &bIsImplemented);
	GX_VERIFY(emStatus);
	if (bIsImplemented)
	{
		emStatus = GXGetFloat(m_hDevice, GX_FLOAT_GAMMA_PARAM, &m_dGamma);
		GX_VERIFY(emStatus);
	}
	else
	{
		m_dGamma = DEFAULT_GAMMA;
	}

	// Check whether the device supports color correction parameters
	emStatus = GXIsImplemented(m_hDevice, GX_INT_COLOR_CORRECTION_PARAM, &bIsImplemented);
	GX_VERIFY(emStatus);
	if (bIsImplemented)
	{
		emStatus = GXGetInt(m_hDevice, GX_INT_COLOR_CORRECTION_PARAM, &m_nColorCorrection);
		GX_VERIFY(emStatus);
	}
	else
	{
		m_nColorCorrection = DEFAULT_COLOR_CORRECTION;
	}	

	do 
	{
		// Get the length of Gamma lookup table
		nStatus = DxGetGammatLut(m_dGamma, NULL, &m_nLutLength);
		if (nStatus != DX_OK)
		{
			break;
		}

		m_pGammaLut = new BYTE[m_nLutLength];
		if (m_pGammaLut == NULL)
		{
			nStatus = DX_NOT_ENOUGH_SYSTEM_MEMORY;
			break;
		}

		// Calculate the Gamma lookup table
		nStatus = DxGetGammatLut(m_dGamma, m_pGammaLut, &m_nLutLength);
		if (nStatus != DX_OK)
		{
			break;
		}

		// Get the length of contrast lookup table 
		nStatus = DxGetContrastLut((int)m_nContrast, NULL, &m_nLutLength);
		if (nStatus != DX_OK)
		{
			break;
		}

		m_pContrastLut = new BYTE[m_nLutLength];
		if (m_pContrastLut == NULL)
		{
			nStatus = DX_NOT_ENOUGH_SYSTEM_MEMORY;
			break;
		}

		// Calculate the contrast lookup table
		nStatus = DxGetContrastLut((int)m_nContrast, m_pContrastLut, &m_nLutLength);
		if (nStatus != DX_OK)
		{
			break;
		}

	} while (0);

	// Fail to set lookup table and release memory
	if (nStatus != DX_OK)
	{
		if (m_pGammaLut != NULL)
		{
			delete[] m_pGammaLut;
			m_pGammaLut = NULL;
		}
		if (m_pContrastLut != NULL)
		{
			delete[] m_pContrastLut;
			m_pContrastLut = NULL;
		}	
		MessageBox("Fail to get image processing LUT parameters!");
		return;
	}
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize the UI

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamColorDlg::InitUI()
{
	GX_STATUS emStatus = GX_STATUS_ERROR;
	int  nValue = 0;

	InitTriggerModeUI();

	InitTriggerSourceUI();

	InitTriggerActivationUI();

    InitEnumUI(GX_ENUM_BALANCE_WHITE_AUTO, (CComboBox *)GetDlgItem(IDC_COMBO_AWB), m_bColorFilter);

	InitEnumUI(GX_ENUM_BALANCE_RATIO_SELECTOR, (CComboBox *)GetDlgItem(IDC_COMBO_RATIO_SELECTOR), m_bBalanceRatioSelector);

	InitShutterUI();

	InitGainUI();

	InitWhiteRatioUI();
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize the BalanceRatio control 

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamColorDlg::InitWhiteRatioUI()
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	CString   strTemp  = _T("");

	if (!m_bBalanceRatioSelector)
	{
		return;
	}

	// Get the range of the ratio
	emStatus = GXGetFloatRange(m_hDevice,GX_FLOAT_BALANCE_RATIO,&m_stFloatRange);
	GX_VERIFY(emStatus);
	strTemp.Format("BalanceRatio(%.4f~%.4f)",m_stFloatRange.dMin,m_stFloatRange.dMax);
	SetDlgItemText(IDC_STATIC_WHITE_RATIO,strTemp);

	// Refresh the BalanceRatio 
	RefreshWhiteRatio();
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to refresh white ratio 

\return  void
*/
//---------------------------------------------------------------------------------
void CGxSingleCamColorDlg::RefreshWhiteRatio()
{
	GX_STATUS emStatus = GX_STATUS_ERROR;
	double    dValue   = 0.0;
	CString   strTemp  = _T("");

	// Get current value of white ratio and update to the UI
	emStatus = GXGetFloat(m_hDevice, GX_FLOAT_BALANCE_RATIO, &dValue);
	GX_VERIFY(emStatus);

	m_dBalanceRatio = dValue;
	strTemp.Format("%.4f", dValue);
	SetDlgItemText(IDC_EDIT_BALANCE_RATIO, strTemp);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize the TriggerMode Combobox 

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamColorDlg::InitTriggerModeUI()
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

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize the TriggerSource Combobox 

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamColorDlg::InitTriggerSourceUI()
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

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize TriggerActivation Combobox 

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamColorDlg::InitTriggerActivationUI()
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
	InitEnumUI(GX_ENUM_TRIGGER_ACTIVATION, (CComboBox *)GetDlgItem(IDC_COMBO_TRIGGER_ACTIVE), m_bTriggerActive);
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
void CGxSingleCamColorDlg::InitEnumUI(GX_FEATURE_ID_CMD emFeatureID, CComboBox *pComboBox, bool bIsImplement)
{
	// Check whether the ComboBox is available
	if ((pComboBox == NULL) || !bIsImplement)
	{
		return;
	}

	GX_ENUM_DESCRIPTION *pEnum       = NULL;
	GX_STATUS emStatus    = GX_STATUS_ERROR;
	size_t    nbufferSize = 0;
	uint32_t  nEntryNum   = 0;
	int64_t   nEnumValue  = -1;
	int       nCursel     = 0;
	double    dValue      = 0;

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
		ShowErrorString(emStatus);
		return;
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

//---------------------------------------------------------------------------------
/**
\brief   This is a callback function that demonstrates how to convert the image format and draw image
\param   Callback parameters

\return  void
*/
//----------------------------------------------------------------------------------
void __stdcall CGxSingleCamColorDlg::OnFrameCallbackFun(GX_FRAME_CALLBACK_PARAM* pFrame)
{
	CGxSingleCamColorDlg *pDlg = (CGxSingleCamColorDlg*)(pFrame->pUserParam);

	if (pFrame->status == 0)
	{
		memcpy(pDlg->m_pBufferRaw, pFrame->pImgBuf, pFrame->nImgSize);

		DxRaw8toRGB24( pDlg->m_pBufferRaw
			         , pDlg->m_pBufferRGB
					 , (VxUint32)(pDlg->m_nImageWidth)
					 , (VxUint32)(pDlg->m_nImageHeight)
					 , RAW2RGB_NEIGHBOUR		         
					 , DX_PIXEL_COLOR_FILTER(pDlg->m_nPixelColorFilter)
					 , TRUE);

		if (pDlg->m_bIsImproveImg)
		{
			DxImageImprovment(pDlg->m_pBufferRGB
				            , pDlg->m_pBufferRGB
							, (VxUint32)(pDlg->m_nImageWidth)
							, (VxUint32)(pDlg->m_nImageHeight)
							, pDlg->m_nColorCorrection
							, pDlg->m_pContrastLut
							, pDlg->m_pGammaLut);
		}

		pDlg->DrawImg();

		if (pDlg->m_bIsSaveImg)
		{
			pDlg->SaveImage();
		}
	}
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to draw image in the window

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamColorDlg::DrawImg()
{	
	int nWndWidth  = 0;
	int nWndHeight = 0;

	// Initialize the parameters for drawing image.
	RECT objRect;
	m_pWnd->GetClientRect(&objRect);	
	nWndWidth  = objRect.right - objRect.left;
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
		            m_pBufferRGB,
		            m_pBmpInfo,
		            DIB_RGB_COLORS,
		            SRCCOPY
		            );
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to allocate memory for showing color image.

\return  true:Success false:Fail
*/
//----------------------------------------------------------------------------------
bool CGxSingleCamColorDlg::PrepareForShowImg()
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
	//-----------------------Allocate buffer for image data----------------
	//Allocate memory for showing RGB images.
	m_pBufferRGB = new BYTE[(size_t)(m_nImageWidth * m_nImageHeight * 3)];
	if (m_pBufferRGB == NULL)
	{
		return false;
	}
    
	//Allocate memory for showing raw image  
	m_pBufferRaw = new BYTE[(size_t)m_nPayLoadSize];
	if (m_pBufferRaw == NULL)
	{
		delete []m_pBufferRGB;
		m_pBufferRGB = NULL;

		return false;
	}

	return true;
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to release memory for showing image 
\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamColorDlg::UnPrepareForShowImg()
{
	if (m_pBufferRGB != NULL)
	{
	    delete []m_pBufferRGB;
	    m_pBufferRGB = NULL;
	}
	if (m_pBufferRaw != NULL)
	{
		delete []m_pBufferRaw;
		m_pBufferRaw = NULL;
	}
}

//----------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to save the BMP images.

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamColorDlg::SaveImage() 
{
	// TODO: Add your control notification handler code here
	DWORD		         dwImageSize = (DWORD)(m_nImageWidth * m_nImageHeight * 3);
	BITMAPFILEHEADER     stBfh	     = {0};
	DWORD		         dwBytesRead = 0;
	SYSTEMTIME           sysTime;
	CString              strFileName = "";

	BOOL bRet = CreateDirectory(m_strFilePath, NULL);
	if (bRet)
	{
		SetFileAttributes(m_strFilePath, FILE_ATTRIBUTE_NORMAL);
	}

	// Get current time as the filename of images
	GetLocalTime(&sysTime);
	strFileName.Format("%s\\%d_%d_%d_%d_%d_%d_%d.bmp",m_strFilePath,
		                                              sysTime.wYear,
												      sysTime.wMonth,
		                                              sysTime.wDay,
												      sysTime.wHour,
												      sysTime.wMinute,
												      sysTime.wSecond,
												      sysTime.wMilliseconds);

	stBfh.bfType	= (WORD)'M' << 8 | 'B';			
	stBfh.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);	
	stBfh.bfSize	= stBfh.bfOffBits + dwImageSize; 

	//Create File                                 
	HANDLE hFile = ::CreateFile(strFileName,
		                        GENERIC_WRITE,
		                        0,
		                        NULL,
		                        CREATE_ALWAYS,														
		                        FILE_ATTRIBUTE_NORMAL,
		                        NULL);
	if (hFile != INVALID_HANDLE_VALUE) 
	{
		::WriteFile(hFile, &stBfh, sizeof(BITMAPFILEHEADER), &dwBytesRead, NULL);
		::WriteFile(hFile, m_pBmpInfo, sizeof(BITMAPINFOHEADER), &dwBytesRead, NULL); 
		::WriteFile(hFile, m_pBufferRGB, dwImageSize, &dwBytesRead, NULL);				
		CloseHandle(hFile);
	}
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to send TriggerSoftware command
\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamColorDlg::OnBnClickedBtnSofttrigger() 
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_ERROR;

	//Send TriggerSoftware command
	emStatus = GXSendCommand(m_hDevice, GX_COMMAND_TRIGGER_SOFTWARE);
    GX_VERIFY(emStatus);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to change the TriggerMode

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamColorDlg::OnCbnSelchangeComboTriggerMode() 
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	bool bIsWritable = TRUE;
	int64_t  nCurrentEnumValue = 0;
	CComboBox *pBox    = (CComboBox *)GetDlgItem(IDC_COMBO_TRIGGER_MODE);

	int      nIndex    = pBox->GetCurSel();                     // Get current selection
	int64_t  nEnumVal  = (int64_t)pBox->GetItemData(nIndex);    // Get the enumeration entry of current selection

	emStatus = GXIsWritable(m_hDevice,GX_ENUM_TRIGGER_MODE,&bIsWritable);
	GX_VERIFY(emStatus);

	emStatus = GXGetEnum(m_hDevice, GX_ENUM_TRIGGER_MODE, &nCurrentEnumValue);
	GX_VERIFY(emStatus);

	if (bIsWritable)
	{

		emStatus = GXSetEnum(m_hDevice, GX_ENUM_TRIGGER_MODE, nEnumVal);
		GX_VERIFY(emStatus);
  
		// Change the TriggerMode 
		m_nTriggerMode = nEnumVal;
	}
	else
	{
		MessageBox("The current state cannot write,Please reset after stop acquisition!");
		if (nCurrentEnumValue != nEnumVal)
		{
			if (GX_TRIGGER_MODE_ON == nEnumVal)
			{
				for (int i = 0;i < pBox->GetCount();i++)
				{
					if (GX_TRIGGER_MODE_OFF == pBox->GetItemData(i))
					{
						pBox->SetCurSel(i);
						break;
					}
				}
			}
			else
			{
				for (int i = 0;i < pBox->GetCount();i++)
				{
					if (GX_TRIGGER_MODE_ON == pBox->GetItemData(i))
					{
						pBox->SetCurSel(i);
						break;
					}
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

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to set the value of BalanceWhiteAuto

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamColorDlg::OnCbnSelchangeComboWriteBlance() 
{
	// TODO: Add your control notification handler code here
	CComboBox *pBox    = (CComboBox *)GetDlgItem(IDC_COMBO_AWB);
	GX_STATUS emStatus  = GX_STATUS_SUCCESS;
	int64_t   nEnumVal  = 0;
	int       nIndex    = 0;

	nIndex    = pBox->GetCurSel();                    // Get current selection       
	nEnumVal  = (int64_t)pBox->GetItemData(nIndex);   // Get the enumeration entry of current selection
  
	//Set the WhiteBalanceRatio
	emStatus = GXSetEnum(m_hDevice, GX_ENUM_BALANCE_WHITE_AUTO, nEnumVal);
	GX_VERIFY(emStatus);
	m_nAutoWhiteBalance = nEnumVal;

	UpDateUI();
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to convert the error code to error description
\param  emErrorStatus  [in]   

\return void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamColorDlg::ShowErrorString(GX_STATUS emErrorStatus)
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
\brief  This function demonstrates how to use the control to update UI 

\return void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamColorDlg::UpDateUI()
{
	GetDlgItem(IDC_BTN_OPEN_DEVICE)->EnableWindow(!m_bDevOpened);
	GetDlgItem(IDC_BTN_CLOSE_DEVICE)->EnableWindow(m_bDevOpened);
	GetDlgItem(IDC_BTN_START_SNAP)->EnableWindow(m_bDevOpened && !m_bIsSnap);
	GetDlgItem(IDC_BTN_STOP_SNAP)->EnableWindow(m_bDevOpened && m_bIsSnap);
	GetDlgItem(IDC_BTN_SOFTTRIGGER)->EnableWindow(m_bDevOpened);

	GetDlgItem(IDC_EDIT_GAIN)->EnableWindow(m_bDevOpened);
	GetDlgItem(IDC_EDIT_SHUTTER)->EnableWindow(m_bDevOpened);

	GetDlgItem(IDC_COMBO_TRIGGER_ACTIVE)->EnableWindow(m_bDevOpened && m_bTriggerActive);
	GetDlgItem(IDC_COMBO_TRIGGER_MODE)->EnableWindow(m_bDevOpened && m_bTriggerMode);
	GetDlgItem(IDC_COMBO_TRIGGER_SOURCE)->EnableWindow(m_bDevOpened && m_bTriggerSource);
	GetDlgItem(IDC_COMBO_AWB)->EnableWindow(m_bDevOpened && m_bColorFilter);
	GetDlgItem(IDC_COMBO_RATIO_SELECTOR)->EnableWindow(m_bDevOpened && m_bBalanceRatioSelector);
    
	GetDlgItem(IDC_EDIT_BALANCE_RATIO)->EnableWindow(m_bDevOpened 
		                                         && (m_nAutoWhiteBalance == GX_BALANCE_WHITE_AUTO_OFF)
												 &&  m_bBalanceRatioSelector);

	GetDlgItem(IDC_CHECK_IMAGE_IMPROVE)->EnableWindow(m_bDevOpened);
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to set the switch for improving the quality of image                         
\return void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamColorDlg::OnBnClickedCheckImageImprove() 
{
	// TODO: Add your control notification handler code here
	m_bIsImproveImg = !m_bIsImproveImg;
}


//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to enumerate all camera devices and initialize device.

\return void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamColorDlg::OnBnClickedBtnOpenDevice()
{
	// TODO: Add your control notification handler code here
	GX_STATUS      emStatus = GX_STATUS_SUCCESS;
	uint32_t       nDevNum  = 0;
	GX_OPEN_PARAM  stOpenParam;
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

	// If the device is opened then close it to ensure the camera could open again.
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

	GetImageImproveParam();

	SetTimer(0, 1000, NULL);

	UpDateUI();
}

//----------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize device

\return  void
*/
//----------------------------------------------------------------------------------
GX_STATUS CGxSingleCamColorDlg::InitDevice()
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

// ---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to set the pixel format to 8bit

\return  emStatus GX_STATUS_SUCCESS: Success, else: Fail
*/
// ----------------------------------------------------------------------------------
GX_STATUS CGxSingleCamColorDlg::SetPixelFormat8bit()
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
\brief  This function demonstrates how to get the parameters of the device

\return void
*/
//----------------------------------------------------------------------------------
GX_STATUS CGxSingleCamColorDlg::GetDeviceParam()
{
	GX_STATUS emStatus       = GX_STATUS_ERROR;
	bool      bColorFliter   = false;

	// Get image size
	emStatus = GXGetInt(m_hDevice, GX_INT_PAYLOAD_SIZE, &m_nPayLoadSize);
	VERIFY_STATUS_RET(emStatus);

	// Get image width
	emStatus = GXGetInt(m_hDevice, GX_INT_WIDTH, &m_nImageWidth);
	VERIFY_STATUS_RET(emStatus);

	// Check whether the device supports PixelColorFilter
	emStatus = GXIsImplemented(m_hDevice, GX_ENUM_PIXEL_COLOR_FILTER, &m_bColorFilter);
	VERIFY_STATUS_RET(emStatus);
	if (m_bColorFilter)
	{
		emStatus = GXGetEnum(m_hDevice, GX_ENUM_BALANCE_WHITE_AUTO, &m_nAutoWhiteBalance);
		VERIFY_STATUS_RET(emStatus);

		emStatus = GXGetEnum(m_hDevice, GX_ENUM_PIXEL_COLOR_FILTER, &m_nPixelColorFilter);
		VERIFY_STATUS_RET(emStatus);
	}

	//Check whether the device supports BalanceRatioSelector 
	emStatus = GXIsImplemented(m_hDevice, GX_ENUM_BALANCE_RATIO_SELECTOR, &m_bBalanceRatioSelector);
	VERIFY_STATUS_RET(emStatus);

	// Get image height
	emStatus = GXGetInt(m_hDevice, GX_INT_HEIGHT, &m_nImageHeight);

	return emStatus;
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to close device.
\return void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamColorDlg::OnBnClickedBtnCloseDevice()
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

	if (m_pGammaLut != NULL)
	{
		delete[] m_pGammaLut;
		m_pGammaLut = NULL;
	}
	if (m_pContrastLut != NULL)
	{
		delete[] m_pContrastLut;
		m_pContrastLut = NULL;
	}	

	KillTimer(0);

	UpDateUI();
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to stop acquisition

\return void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamColorDlg::OnBnClickedBtnStopSnap()
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

	UpDateUI();
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to start acquisition

\return void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamColorDlg::OnBnClickedBtnStartSnap()
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

	//Send AquisitionStart command
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
\brief  This function demonstrates how to change the TriggerResource

\return void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamColorDlg::OnCbnSelchangeComboTriggerSource()
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	CComboBox *pBox    = (CComboBox *)GetDlgItem(IDC_COMBO_TRIGGER_SOURCE);

	int      nIndex    = pBox->GetCurSel();                   // Get current selection
	int64_t  nEnumVal  = (int64_t)pBox->GetItemData(nIndex);  // Get the enumeration entry of current selection
	emStatus = GXSetEnum(m_hDevice, GX_ENUM_TRIGGER_SOURCE, nEnumVal);
	GX_VERIFY(emStatus);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to change the TriggerActivation

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamColorDlg::OnCbnSelchangeComboTriggerActive()
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	CComboBox *pBox    = (CComboBox *)GetDlgItem(IDC_COMBO_TRIGGER_ACTIVE);

	int      nIndex    = pBox->GetCurSel();                   // Get current selection
	int64_t  nEnumVal  = (int64_t)pBox->GetItemData(nIndex);  // Get the enumeration entry of current selection

	emStatus = GXSetEnum(m_hDevice, GX_ENUM_TRIGGER_ACTIVATION, nEnumVal);
	GX_VERIFY(emStatus);
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to set the switch for saving the BMP images

\return void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamColorDlg::OnBnClickedCheckSaveBmp()
{
	// TODO: Add your control notification handler code here
	m_bIsSaveImg = !m_bIsSaveImg;
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize the Gain control 
 
\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamColorDlg::InitGainUI()
{
	CStatic     *pStcGain     = (CStatic *)GetDlgItem(IDC_STC_GAIN);
	CEdit       *pEditGainVal = (CEdit *)GetDlgItem(IDC_EDIT_GAIN);
	int          nPos         = 0;

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

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize the ExposureTime control

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamColorDlg::InitShutterUI()
{
	CStatic     *pStcShutterShow = (CStatic *)GetDlgItem(IDC_STC_SHUTTER_SHOW);
	CEdit       *pEditShutterVal = (CEdit *)GetDlgItem(IDC_EDIT_SHUTTER);
	int          nPos            = 0;

	// Check whether the edit control is available
	if ((pEditShutterVal == NULL) || (pStcShutterShow == NULL))
	{
		return;
	}

	GX_STATUS      emStatus = GX_STATUS_ERROR;
	CString         strTemp = "";
	double    dShutterValue = 0.0;

	// Get the range of ExposureTime value
	emStatus = GXGetFloatRange(m_hDevice, GX_FLOAT_EXPOSURE_TIME, &m_stShutterFloatRange);
	GX_VERIFY(emStatus);

	strTemp.Format("ExposureTime(%.4f~%.4f)%s", m_stShutterFloatRange.dMin, m_stShutterFloatRange.dMax, m_stShutterFloatRange.szUnit);
	pStcShutterShow->SetWindowText(strTemp);
	
	// Get current value of ExposureTime and update to the UI
	emStatus = GXGetFloat(m_hDevice, GX_FLOAT_EXPOSURE_TIME, &dShutterValue);
	GX_VERIFY(emStatus);
	
	m_dShutterValue = dShutterValue;

	UpdateData(FALSE);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to set the value of BalanceRatioSelector
\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamColorDlg::OnCbnSelchangeComboRatioSelector()
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	CComboBox *pBox    = (CComboBox *)GetDlgItem(IDC_COMBO_RATIO_SELECTOR);
	double     dValue  = 0.0;

	int      nIndex    = pBox->GetCurSel();                   // Get current option
	int64_t  nEnumVal  = (int64_t)pBox->GetItemData(nIndex);  // Get the enumeration entry of current selection

	emStatus = GXSetEnum(m_hDevice, GX_ENUM_BALANCE_RATIO_SELECTOR, nEnumVal);
	GX_VERIFY(emStatus);

	InitWhiteRatioUI();
}

//---------------------------------------------------------------------------------
/**
\brief   "BalanceRatio"Edit box lost focus response function

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamColorDlg::OnEnKillfocusEditBalanceRatio()
{
	// TODO: Add your control notification handler code here
	//Judge validity of handle,avoid exception due to closing program after device off-line
	if (m_hDevice == NULL)
	{
		return;
	}

	UpdateData(TRUE);
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	//If greater than Max then set BalanceRatio as Max 
	if (m_dBalanceRatio > m_stFloatRange.dMax)
	{
		m_dBalanceRatio = m_stFloatRange.dMax;
	}
	//If less than Min then set BalanceRatio as Min
	if (m_dBalanceRatio < m_stFloatRange.dMin)
	{
		m_dBalanceRatio = m_stFloatRange.dMin;
	}

	emStatus = GXSetFloat(m_hDevice,GX_FLOAT_BALANCE_RATIO,m_dBalanceRatio);
	GX_VERIFY(emStatus);

    UpdateData(FALSE);
}


//-----------------------------------------------------------------------
/**
\brief   This function demonstrates how to deal with message of "Enter" 
\param   pMsg 

\return  Success: TRUE Fail:FALSE
*/
//-----------------------------------------------------------------------
BOOL CGxSingleCamColorDlg::PreTranslateMessage(MSG* pMsg)
{
	// TODO: Add your specialized code here and/or call the base class
	CWnd  *pWnd   = NULL;
	int   nCtrlID = 0;                      

	
	if ((pMsg->message == WM_KEYDOWN) && (pMsg->wParam == VK_RETURN))   
	{    
	
		pWnd = GetFocus();

		
		nCtrlID = pWnd->GetDlgCtrlID();

		switch(nCtrlID)
		{
		case IDC_EDIT_BALANCE_RATIO:
		case IDC_EDIT_SHUTTER:
		case IDC_EDIT_GAIN:

		
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

//-----------------------------------------------------------------------
/**
\brief    This function demonstrates how to set the ExposureTime

\return   void
*/
//-----------------------------------------------------------------------
void CGxSingleCamColorDlg::OnKillfocusEditShutter() 
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

//-----------------------------------------------------------------------
/**
\brief   This function demonstrates how to set the Gain

\return  void
*/
//-----------------------------------------------------------------------
void CGxSingleCamColorDlg::OnKillfocusEditGain()
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


//----------------------------------------------------------------
/**
\brief  This timer callback function demonstrates how to update the value of BalanceWhiteAuto to UI
\param                                          

\return void
*/
//----------------------------------------------------------------
void CGxSingleCamColorDlg::OnTimer(UINT_PTR nIDEvent) 
{
	// TODO: Add your message handler code here and/or call default
	GX_STATUS emStatus = GX_STATUS_ERROR;

	//Get the current value of BalanceWhiteAuto from the device and update to UI
	if (m_nAutoWhiteBalance == GX_BALANCE_WHITE_AUTO_ONCE)
	{
		int64_t nWBEnumVal = 0;   

		CComboBox *pAutoWBBox = (CComboBox *)GetDlgItem(IDC_COMBO_AWB);

		//Get the enumeration entry of BalanceWhiteAuto
		emStatus = GXGetEnum(m_hDevice, GX_ENUM_BALANCE_WHITE_AUTO, &nWBEnumVal);
		if (emStatus != GX_STATUS_SUCCESS)
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
					pAutoWBBox->SetCurSel(i);
					break;
				}
			}
			UpDateUI();	
			RefreshWhiteRatio();
		}
	}

	// Check whether the BalanceWhiteAuto is Continuous
	if (m_nAutoWhiteBalance == GX_BALANCE_WHITE_AUTO_CONTINUOUS)
	{
		RefreshWhiteRatio();
	}

	CDialog::OnTimer(nIDEvent);
}
