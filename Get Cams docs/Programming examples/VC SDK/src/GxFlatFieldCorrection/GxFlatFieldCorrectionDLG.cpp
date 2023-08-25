// GxFlatFieldCorrectionDlg.cpp : implementation file
//

#include "stdafx.h"
#include "GxFlatFieldCorrection.h"
#include "GxFlatFieldCorrectionDlg.h" 

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

#undef new

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
// CGxFlatFieldCorrectionDlg dialog

CGxFlatFieldCorrectionDlg::CGxFlatFieldCorrectionDlg(CWnd* pParent /*=NULL*/)
	: CDialog(CGxFlatFieldCorrectionDlg::IDD, pParent)
	, m_bDevOpened(false)
	, m_bIsSnap(false)
	, m_bIsFFC(false)
	, m_bShowFFCOK(false)
	, m_bColorFilter(false)
	, m_bBalanceRatioSelector(false)
	, m_hDevice(NULL)
	, m_pBmpInfo(NULL)
	, m_pRawBuffer(NULL)
	, m_pImgBuffer(NULL)
	, m_pBrightBuf(NULL)
	, m_pDarkBuf(NULL)
	, m_nPayLoadSize(0)
	, m_nImageHeight(0)
	, m_nImageWidth(0)
	, m_nPixelColorFilter(GX_COLOR_FILTER_NONE)
	, m_nActualBits(DX_ACTUAL_BITS_8)
	, m_nRatioSelector(-1)
	, m_nAutoWhiteBalance(0)
	, m_pWnd(NULL)
	, m_hDC(NULL)
	, m_nNewGray(0)
	, m_nFrameCount(CORRECTION_ACTUAL_FRAME_COUNT_4)
	, m_bDarkImgAcquired(false)
	, m_bBrightImgAcquired(false)
	, m_bExecuteFFC(false)
	, m_bGetFFCCoefficients(false)
	, m_nBrightImgNum(0)
	, m_pAccumulateImgBuf(NULL)
	, m_bBrightImgAcquiredOnce(false)
	, m_bDarkImgAcquiredOnce(false)
	, m_pFFCCoefficients(NULL)
	, m_nFFClength(0)
	, m_nFFCGray(0)
	, m_bFFCGray(false)
	, m_nAverageGrayDarkImg(0)
	, m_nAverageGrayBrightImg(0)
	, m_pAverageImgBuf(NULL)
{
	//{{AFX_DATA_INIT(CGxFlatFieldCorrectionDlg)
	m_dBalanceRatio = 0.0;
	m_dGainValue    = 0.0;
	m_dShutterValue = 0.0;
	//}}AFX_DATA_INIT
	// Note that LoadIcon does not require a subsequent DestroyIcon in Win32
	m_hIcon    = AfxGetApp()->LoadIcon(IDR_MAINFRAME);

	memset(&m_stGainFloatRange, 0, sizeof(GX_FLOAT_RANGE));
	memset(&m_stShutterFloatRange, 0, sizeof(GX_FLOAT_RANGE));
	memset(&m_stFloatRange, 0, sizeof(GX_FLOAT_RANGE));
	memset(m_chBmpBuf, 0, sizeof(m_chBmpBuf));
	memset(&m_stFlatFieldCorrection, 0, sizeof(FLAT_FIELD_CORRECTION_PROCESS));
}

void CGxFlatFieldCorrectionDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CGxFlatFieldCorrectionDlg)
	DDX_Text(pDX, IDC_EDIT_BALANCE_RATIO, m_dBalanceRatio);
	DDX_Text(pDX, IDC_EDIT_GAIN, m_dGainValue);
	DDX_Text(pDX, IDC_EDIT_SHUTTER, m_dShutterValue);
	//}}AFX_DATA_MAP
	DDX_Control(pDX, IDC_SLIDER_FRAME_NUM, m_sliderFrameCount);
	DDX_Text(pDX, IDC_EDIT_GRAY, m_nNewGray);
	DDX_Control(pDX, IDC_SLIDER_GRAY, m_sliderGrayValue);
}

BEGIN_MESSAGE_MAP(CGxFlatFieldCorrectionDlg, CDialog)
	//{{AFX_MSG_MAP(CGxFlatFieldCorrectionDlg)
	ON_WM_QUERYDRAGICON()
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_CLOSE()
	ON_CBN_SELCHANGE(IDC_COMBO_AWB, OnCbnSelchangeComboWriteBlance)
    ON_BN_CLICKED(IDC_BTN_OPEN_DEVICE, OnBnClickedBtnOpenDevice)
    ON_BN_CLICKED(IDC_BTN_CLOSE_DEVICE, OnBnClickedBtnCloseDevice)
    ON_BN_CLICKED(IDC_BTN_STOP_SNAP, OnBnClickedBtnStopSnap)
    ON_BN_CLICKED(IDC_BTN_START_SNAP, OnBnClickedBtnStartSnap)
    ON_CBN_SELCHANGE(IDC_COMBO_RATIO_SELECTOR, OnCbnSelchangeComboRatioSelector)
	ON_EN_KILLFOCUS(IDC_EDIT_BALANCE_RATIO, OnEnKillfocusEditBalanceRatio)
	ON_EN_KILLFOCUS(IDC_EDIT_SHUTTER, OnKillfocusEditShutter)
	ON_EN_KILLFOCUS(IDC_EDIT_GAIN, OnKillfocusEditGain)
	ON_WM_TIMER()
	//}}AFX_MSG_MAP
	
	ON_WM_HSCROLL()
	ON_EN_KILLFOCUS(IDC_EDIT_GRAY, OnKillfocusEditGray)
	ON_BN_CLICKED(IDC_BUTTON_ACQUIRE_DARK_IMG, &CGxFlatFieldCorrectionDlg::OnBnClickedButtonAcquireDarkImg)
	ON_BN_CLICKED(IDC_BUTTON_ACQUIRE_BRIGHT_IMG, &CGxFlatFieldCorrectionDlg::OnBnClickedButtonAcquireBrightImg)
	ON_BN_CLICKED(IDC_BUTTON_EXECUTE_FFC, &CGxFlatFieldCorrectionDlg::OnBnClickedButtonExecuteFFC)
	ON_BN_CLICKED(IDC_CHECK_PREVIEW, &CGxFlatFieldCorrectionDlg::OnBnClickedCheckPreview)
	ON_BN_CLICKED(IDC_CHECK_GRAY, &CGxFlatFieldCorrectionDlg::OnBnClickedCheckGray)
	ON_BN_CLICKED(IDC_BUTTON_READ, &CGxFlatFieldCorrectionDlg::OnBnClickedButtonRead)
	ON_BN_CLICKED(IDC_BUTTON_WRITE, &CGxFlatFieldCorrectionDlg::OnBnClickedButtonWrite)
	ON_NOTIFY(UDN_DELTAPOS, IDC_SPIN_EXPECT_GRAY_VALUE, &CGxFlatFieldCorrectionDlg::OnDeltaposSpinExpectGrayValue)
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CGxFlatFieldCorrectionDlg message handlers

BOOL CGxFlatFieldCorrectionDlg::OnInitDialog()
{
	CDialog::OnInitDialog();

	// Add "About..." menu item to system menu.

	// IDM_ABOUTBOX must be in the system command range
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
	//  when the application's main window is not a dialog.
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon
	
	// TODO: Add extra initialization here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	// Before using any GxIAPI methods, the GxIAPI must be initialized
	emStatus = GXInitLib();
	if (emStatus != GX_STATUS_SUCCESS)
	{
		ShowErrorString(emStatus);
		exit(0); 
	}

	//Get the pointer and DC handle for showing image
	m_pWnd = GetDlgItem(IDC_STATIC_SHOW_FRAME);
	m_hDC  = ::GetDC(m_pWnd->m_hWnd);

	UpDateUI();
	return TRUE;  // return TRUE  unless you set the focus to a control
}

void CGxFlatFieldCorrectionDlg::OnSysCommand(UINT nID, LPARAM lParam)
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

void CGxFlatFieldCorrectionDlg::OnPaint() 
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
HCURSOR CGxFlatFieldCorrectionDlg::OnQueryDragIcon()
{
	return (HCURSOR) m_hIcon;
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to close the application

\return  void
*/
//----------------------------------------------------------------------------------
void CGxFlatFieldCorrectionDlg::OnClose() 
{
	// TODO: Add your message handler code here and/or call default
	KillTimer(0);

	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	// If the device is acquiring image then stop it
	if (m_bIsSnap)
	{
		// Send AcquisitionStop command
		emStatus = GXSendCommand(m_hDevice, GX_COMMAND_ACQUISITION_STOP);

		// Unregister the CaptureCallback function
		emStatus = GXUnregisterCaptureCallback(m_hDevice);
		m_bIsSnap = false;
	}

	::ReleaseDC(m_pWnd->m_hWnd, m_hDC);

	// If the device is opened then close it
	if (m_bDevOpened)
	{
		emStatus = GXCloseDevice(m_hDevice);
		m_bDevOpened = false;
		m_hDevice    = NULL;
	}

	// Release memory for showing image
	UnPrepareForShowImg();

	// Release GxIAPI
	emStatus = GXCloseLib();

	CDialog::OnClose();
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize the UI 

\return  void
*/
//----------------------------------------------------------------------------------
void CGxFlatFieldCorrectionDlg::InitUI()
{
	GX_STATUS emStatus = GX_STATUS_ERROR;
	int  nValue = 0;

    InitEnumUI(GX_ENUM_BALANCE_WHITE_AUTO, (CComboBox *)GetDlgItem(IDC_COMBO_AWB), m_bColorFilter);

	InitEnumUI(GX_ENUM_BALANCE_RATIO_SELECTOR, (CComboBox *)GetDlgItem(IDC_COMBO_RATIO_SELECTOR), m_bBalanceRatioSelector);

	InitShutterUI();

	InitGainUI();

	InitWhiteRatioUI();

	InitGrayUI();

	InitFrameCountUI();
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize the BalanceRatio control 

\return  void
*/
//----------------------------------------------------------------------------------
void CGxFlatFieldCorrectionDlg::InitWhiteRatioUI()
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	CString   strTemp  = _T("");

	if (!m_bBalanceRatioSelector)
	{
		return;
	}

	// Get the range of the ratio
	emStatus = GXGetFloatRange(m_hDevice, GX_FLOAT_BALANCE_RATIO, &m_stFloatRange);
    GX_VERIFY(emStatus);
	strTemp.Format("BalanceRatio(%.4f~%.4f)", m_stFloatRange.dMin, m_stFloatRange.dMax);
	SetDlgItemText(IDC_STATIC_WHITE_RATIO, strTemp);

	// Refresh the BalanceRatio 
	RefreshWhiteRatio();
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
void CGxFlatFieldCorrectionDlg::InitEnumUI(GX_FEATURE_ID_CMD emFeatureID, CComboBox *pComboBox, bool bIsImplement)
{
	// Check whether the ComboBox is available
	if ((pComboBox == NULL) || !bIsImplement)
	{
		return;
	}

	GX_ENUM_DESCRIPTION *pEnum = NULL;
	GX_STATUS emStatus    = GX_STATUS_ERROR;
	size_t    nbufferSize =  0;
	uint32_t  nEntryNum   =  0;
	int64_t   nEnumValue  = -1;
	int       nCursel     =  0;
	double    dValue      =  0;

	// Get the enumeration entry the device supports
	emStatus = GXGetEnumEntryNums(m_hDevice, emFeatureID, &nEntryNum);
	GX_VERIFY(emStatus);

	// Allocate memory for getting the description of the enumeration entry
	nbufferSize = nEntryNum * sizeof(GX_ENUM_DESCRIPTION);
	pEnum       = new(std::nothrow) GX_ENUM_DESCRIPTION[nEntryNum];
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
\param   pFrame    Callback parameters

\return  void
*/
//----------------------------------------------------------------------------------
void __stdcall CGxFlatFieldCorrectionDlg::OnFrameCallbackFun(GX_FRAME_CALLBACK_PARAM* pFrame)
{
	CGxFlatFieldCorrectionDlg *pDlg        = (CGxFlatFieldCorrectionDlg*)(pFrame->pUserParam);
	VxUint32                  nImageHeight = (VxUint32)pDlg->m_nImageHeight;
	VxUint32                  nImageWidth  = (VxUint32)pDlg->m_nImageWidth;
	VxInt32                   DxStatus     = DX_OK;

	// Check the status of the current image
	if (pFrame->status != 0)
	{
		return;
	}

	// Acquire dark field image
	if (true == pDlg->m_bDarkImgAcquired)
	{
		memcpy(pDlg->m_pDarkBuf, pFrame->pImgBuf, pFrame->nImgSize);
		pDlg->AcquireDarkFieldImg();
		pDlg->m_bDarkImgAcquired = false;
	}

	// Acquire bright field image
	if (true == pDlg->m_bBrightImgAcquired)
	{
		memcpy(pDlg->m_pBrightBuf, pFrame->pImgBuf, pFrame->nImgSize);
		pDlg->AcquireBrightFieldImg();
	}

	memcpy(pDlg->m_pRawBuffer, pFrame->pImgBuf, pFrame->nImgSize);

	// Check whether to perform flat field correction
	if (pDlg->m_bIsFFC)
	{
		DxStatus = DxFlatFieldCorrection(pDlg->m_pRawBuffer
						               , pDlg->m_pRawBuffer
			                           , pDlg->m_nActualBits
                                       , (VxUint32)(pDlg->m_nImageWidth)
			                           , (VxUint32)(pDlg->m_nImageHeight)
			                           , pDlg->m_pFFCCoefficients
			                           , &pDlg->m_nFFClength);
		if (DX_OK != DxStatus)
		{
			pDlg->m_bIsFFC = false;
			pDlg->GetDlgItem(IDC_STATIC_PROMPT_INFOR)->SetWindowText("The application of flat field correction coefficients fails!");
			pDlg->UpDateUI();
		}
		else
		{
			if (!pDlg->m_bShowFFCOK)
			{
				pDlg->GetDlgItem(IDC_STATIC_PROMPT_INFOR)->SetWindowText("The application of flat field correction coefficients succeeds!");  
				pDlg->UpDateUI();
				pDlg->m_bShowFFCOK = true;
			}
		}
	}

	// If the acquired image is color format, convert it to RGB
	if (pDlg->m_bColorFilter)
	{
		DxStatus = DxRaw8toRGB24(pDlg->m_pRawBuffer
			                   , pDlg->m_pImgBuffer
			                   , (VxUint32)(pDlg->m_nImageWidth)
			                   , (VxUint32)(pDlg->m_nImageHeight)
			                   , RAW2RGB_NEIGHBOUR		         
			                   , DX_PIXEL_COLOR_FILTER(pDlg->m_nPixelColorFilter)
			                   , TRUE);
		if (DX_OK != DxStatus)
		{
			return;
		}
	} 
	else
	{
		// If the acquired image is mono format, you must flip the image data for showing
		for(VxUint32 i = 0; i < nImageHeight; i++)
		{
			memcpy(pDlg->m_pImgBuffer + i * nImageWidth, pDlg->m_pRawBuffer + (nImageHeight - i - 1) * nImageWidth, (size_t)nImageWidth);
		}
	}

	// Show the processed image.
	pDlg->DrawImg();
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to draw image in the window

\return  void
*/
//----------------------------------------------------------------------------------
void CGxFlatFieldCorrectionDlg::DrawImg()
{	
	int nWndWidth  = 0;
	int nWndHeight = 0;

	// Initialize the parameters for drawing image
	RECT objRect;
	m_pWnd->GetClientRect(&objRect);	
	nWndWidth  = objRect.right - objRect.left;
	nWndHeight = objRect.bottom - objRect.top;

	// You could use the following code to show image
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
		            m_pImgBuffer,
		            m_pBmpInfo,
		            DIB_RGB_COLORS,
		            SRCCOPY
		            );
}

//----------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to allocate memory for showing color image

\return  true:Success false:Fail
*/
//----------------------------------------------------------------------------------
bool CGxFlatFieldCorrectionDlg::PrepareForShowColorImg()
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
	//Allocate memory for showing RGB images
	m_pImgBuffer = new(std::nothrow) BYTE[(size_t)(m_nImageWidth * m_nImageHeight * 3)];
	if (m_pImgBuffer == NULL)
	{
		return false;
	}

	return true;
}

//----------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to allocate memory for showing mono image

\return  true:Success false:Fail
*/
//----------------------------------------------------------------------------------
bool CGxFlatFieldCorrectionDlg::PrepareForShowMonoImg()
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

	// If using mono cameras, you need to initialize color palette first
	for(int i = 0; i < 256; i++)
	{
		m_pBmpInfo->bmiColors[i].rgbBlue	 = i;
		m_pBmpInfo->bmiColors[i].rgbGreen	 = i;
		m_pBmpInfo->bmiColors[i].rgbRed		 = i;
		m_pBmpInfo->bmiColors[i].rgbReserved = i;
	}

	//---------------------------------------------------------------------
	//-------------------Allocate buffer for showing image data------------
	// Allocate memory for showing mono images.
	m_pImgBuffer = new(std::nothrow) BYTE[(size_t)(m_nImageWidth * m_nImageHeight)];
	if (m_pImgBuffer == NULL)
	{
		return false;
	}

	return true;
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to allocate memory for showing image

\return  true:Success false:Fail
*/
//----------------------------------------------------------------------------------
bool CGxFlatFieldCorrectionDlg::PrepareForShowImg()
{
	bool bRet = true;

	do 
	{
		// Allocate memory for showing raw image
		m_pRawBuffer = new(std::nothrow) BYTE[(size_t)(m_nImageWidth * m_nImageHeight)];
		if (m_pRawBuffer == NULL)
		{
			bRet = false;
			break;
		}

		// Allocate memory for the data of dark field image
		m_pDarkBuf = new(std::nothrow) BYTE[(size_t)(m_nImageWidth * m_nImageHeight)];
		if (m_pDarkBuf == NULL)
		{
			bRet = false;
			break;
		}
		memset(m_pDarkBuf, 0, (size_t)(m_nImageWidth * m_nImageHeight));

		// Allocate memory for the data of bright field image
		m_pBrightBuf = new(std::nothrow) BYTE[(size_t)(m_nImageWidth * m_nImageHeight)];
		if (m_pBrightBuf == NULL)
		{
			bRet = false;
			break;
		}

		// Allocate memory for the data of accumulated bright field image
		m_pAccumulateImgBuf = new(std::nothrow) VxUint32[(size_t)(m_nImageWidth * m_nImageHeight)];
		if (m_pAccumulateImgBuf == NULL)
		{
			bRet = false;
			break;
		}

		/// Allocate memory for the data of average bright field image
		m_pAverageImgBuf = new(std::nothrow) BYTE[(size_t)(m_nImageWidth * m_nImageHeight)];
		if (m_pAverageImgBuf == NULL)
		{
			bRet = false;
			break;
		}

		// Get the length of flat field correction(FFC) coefficients
		// Allocate memory for the data of flat field correction coefficients
		m_stFlatFieldCorrection.pBrightBuf = m_pAverageImgBuf;
		m_stFlatFieldCorrection.pDarkBuf   = m_pDarkBuf;
		DxGetFFCCoefficients(m_stFlatFieldCorrection, NULL, &m_nFFClength);

		m_pFFCCoefficients = new(std::nothrow) BYTE[m_nFFClength];
		if (m_pFFCCoefficients == NULL)
		{
			bRet = false;
			break;
		}

		if (m_bColorFilter)   
		{
			// Allocate memory for showing color images
			bRet = PrepareForShowColorImg();
		}
		else  
		{
			// Allocate memory for showing mono images
			bRet = PrepareForShowMonoImg();
		}
	} while (0);


	// If fail to allocate one of the buffers then release buffers have been allocated
	if (!bRet)
	{
		UnPrepareForShowImg();
	}

	return bRet;
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to release memory for showing image

\return  void
*/
//----------------------------------------------------------------------------------
void CGxFlatFieldCorrectionDlg::UnPrepareForShowImg()
{
	if (m_pImgBuffer != NULL)
	{
	    delete []m_pImgBuffer;
	    m_pImgBuffer = NULL;
	}
	if (m_pRawBuffer != NULL)
	{
		delete []m_pRawBuffer;
		m_pRawBuffer = NULL;
	}
	if (m_pDarkBuf != NULL)
	{
		delete []m_pDarkBuf;
		m_pDarkBuf = NULL;
	}
	if (m_pBrightBuf != NULL)
	{
		delete []m_pBrightBuf;
		m_pBrightBuf = NULL;
	}
	if(m_pAccumulateImgBuf != NULL)
	{
		delete []m_pAccumulateImgBuf;
		m_pAccumulateImgBuf = NULL;
	}
	if(m_pAverageImgBuf != NULL)
	{
		delete []m_pAverageImgBuf;
		m_pAverageImgBuf = NULL;
	}
	if(m_pFFCCoefficients != NULL)
	{
		delete []m_pFFCCoefficients;
		m_pFFCCoefficients = NULL;
	}
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to set the value of BalanceWhiteAuto

\return  void
*/
//----------------------------------------------------------------------------------
void CGxFlatFieldCorrectionDlg::OnCbnSelchangeComboWriteBlance() 
{
	// TODO: Add your control notification handler code here
	CComboBox *pBox     = (CComboBox *)GetDlgItem(IDC_COMBO_AWB);
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
\brief   This function demonstrates how to convert the error code to error description
\param   emErrorStatus  [in]   Error information

\return  void
*/
//----------------------------------------------------------------------------------
void CGxFlatFieldCorrectionDlg::ShowErrorString(GX_STATUS emErrorStatus)
{
	char*     pchErrorInfo = NULL;
	size_t    nSize        = 0;
	GX_STATUS emStatus     = GX_STATUS_ERROR;

	// Get the length of error information and allocate memory
	emStatus = GXGetLastError(&emErrorStatus, NULL, &nSize);
	pchErrorInfo = new(std::nothrow) char[nSize];
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
\brief   This function demonstrates how to use the control to update UI

\return  void
*/
//----------------------------------------------------------------------------------
void CGxFlatFieldCorrectionDlg::UpDateUI()
{
	GetDlgItem(IDC_BTN_OPEN_DEVICE)->EnableWindow(!m_bDevOpened);
	GetDlgItem(IDC_BTN_CLOSE_DEVICE)->EnableWindow(m_bDevOpened);
	GetDlgItem(IDC_BTN_START_SNAP)->EnableWindow(m_bDevOpened && !m_bIsSnap);
	GetDlgItem(IDC_BTN_STOP_SNAP)->EnableWindow(m_bDevOpened && m_bIsSnap);

	GetDlgItem(IDC_EDIT_GAIN)->EnableWindow(m_bDevOpened);
	GetDlgItem(IDC_EDIT_SHUTTER)->EnableWindow(m_bDevOpened);

	GetDlgItem(IDC_COMBO_AWB)->EnableWindow(m_bDevOpened && m_bColorFilter);
	GetDlgItem(IDC_COMBO_RATIO_SELECTOR)->EnableWindow(m_bDevOpened && m_bBalanceRatioSelector);
    
	GetDlgItem(IDC_EDIT_BALANCE_RATIO)->EnableWindow(m_bDevOpened 
		                                         && (m_nAutoWhiteBalance == GX_BALANCE_WHITE_AUTO_OFF)
												 &&  m_bBalanceRatioSelector);


	GetDlgItem(IDC_SLIDER_FRAME_NUM)->EnableWindow(m_bDevOpened);
	GetDlgItem(IDC_CHECK_GRAY)->EnableWindow(m_bDevOpened);               
	((CButton*)GetDlgItem(IDC_CHECK_GRAY))->SetCheck(m_bDevOpened && m_bFFCGray);

	GetDlgItem(IDC_SLIDER_GRAY)->EnableWindow(m_bDevOpened && m_bFFCGray);
	GetDlgItem(IDC_EDIT_GRAY)->EnableWindow(m_bDevOpened && m_bFFCGray);
	GetDlgItem(IDC_SPIN_EXPECT_GRAY_VALUE)->EnableWindow(m_bDevOpened && m_bFFCGray);

	GetDlgItem(IDC_BUTTON_ACQUIRE_DARK_IMG)->EnableWindow(m_bDevOpened && m_bIsSnap);
	GetDlgItem(IDC_BUTTON_ACQUIRE_BRIGHT_IMG)->EnableWindow(m_bDevOpened && m_bIsSnap);
	GetDlgItem(IDC_BUTTON_EXECUTE_FFC)->EnableWindow(m_bDevOpened && m_bIsSnap && m_bBrightImgAcquiredOnce);

	GetDlgItem(IDC_CHECK_PREVIEW)->EnableWindow(m_bDevOpened && m_bIsSnap && m_bExecuteFFC);           
	((CButton*)GetDlgItem(IDC_CHECK_PREVIEW))->SetCheck(m_bDevOpened && m_bIsSnap && m_bExecuteFFC && m_bIsFFC);  

	GetDlgItem(IDC_BUTTON_READ)->EnableWindow(m_bDevOpened && m_bIsSnap);                              
	GetDlgItem(IDC_BUTTON_WRITE)->EnableWindow(m_bDevOpened && m_bIsSnap);                            
}

//----------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to enumerate all camera devices and initialize device

\return  void
*/
//----------------------------------------------------------------------------------
void CGxFlatFieldCorrectionDlg::OnBnClickedBtnOpenDevice()
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

	// If the device is opened then close it to ensure the camera could open again
	if (m_hDevice != NULL)
	{
		emStatus = GXCloseDevice(m_hDevice);
		GX_VERIFY(emStatus);
		m_hDevice = NULL;
	}

	//Open the first found device.
	emStatus = GXOpenDevice(&stOpenParam, &m_hDevice);
	GX_VERIFY(emStatus);
	m_bDevOpened = true;

	emStatus = InitDevice();
	GX_VERIFY(emStatus);

	emStatus = GetDeviceParam();
	GX_VERIFY(emStatus);

	InitUI();

	SetTimer(0, 1000, NULL);

	if (!PrepareForShowImg())
	{
		MessageBox("Fail to allocate resource for showing image!");
		return;
	}

	UpDateUI();
}

//----------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize device

\return  void
*/
//----------------------------------------------------------------------------------
GX_STATUS CGxFlatFieldCorrectionDlg::InitDevice()
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	//Set the continuous frame acquisition mode
	emStatus = GXSetEnum(m_hDevice, GX_ENUM_ACQUISITION_MODE, GX_ACQ_MODE_CONTINUOUS);
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
GX_STATUS CGxFlatFieldCorrectionDlg::SetPixelFormat8bit()
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

	// If the PixelSize is 8bit then return, or set the PixelSize to 8bit
	if (nPixelSize == GX_PIXEL_SIZE_BPP8)
	{
		return GX_STATUS_SUCCESS;
	}
	else
	{
		// Get the enumeration entry of the pixel format the device supports
		emStatus = GXGetEnumEntryNums(m_hDevice, GX_ENUM_PIXEL_FORMAT, &nEnmuEntry);
		VERIFY_STATUS_RET(emStatus);

		// Allocate memory for getting the enumeration entry of the pixel format
		nBufferSize      = nEnmuEntry * sizeof(GX_ENUM_DESCRIPTION);
		pEnumDescription = new(std::nothrow) GX_ENUM_DESCRIPTION[nEnmuEntry];

		emStatus = GXGetEnumDescription(m_hDevice, GX_ENUM_PIXEL_FORMAT, pEnumDescription, &nBufferSize);
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
\brief   This function demonstrates how to get the parameters of the device

\return  void
*/
//----------------------------------------------------------------------------------
GX_STATUS CGxFlatFieldCorrectionDlg::GetDeviceParam()
{
	GX_STATUS emStatus       = GX_STATUS_ERROR;
	bool      bColorFliter   = false;

	// Get image size
	emStatus = GXGetInt(m_hDevice, GX_INT_PAYLOAD_SIZE, &m_nPayLoadSize);
	VERIFY_STATUS_RET(emStatus);

	// Get image width
	emStatus = GXGetInt(m_hDevice, GX_INT_WIDTH, &m_nImageWidth);
	VERIFY_STATUS_RET(emStatus);
	m_stFlatFieldCorrection.nImgWid = (VxUint32)m_nImageWidth;
	
	// Check whether the device supports PixelColorFilter
	emStatus = GXIsImplemented(m_hDevice, GX_ENUM_PIXEL_COLOR_FILTER, &m_bColorFilter);
	VERIFY_STATUS_RET(emStatus);
	if (m_bColorFilter)
	{
		emStatus = GXGetEnum(m_hDevice, GX_ENUM_BALANCE_WHITE_AUTO, &m_nAutoWhiteBalance);
		VERIFY_STATUS_RET(emStatus);

		emStatus = GXGetEnum(m_hDevice, GX_ENUM_PIXEL_COLOR_FILTER, &m_nPixelColorFilter);
		VERIFY_STATUS_RET(emStatus);

		m_stFlatFieldCorrection.emBayerType = (DX_PIXEL_COLOR_FILTER)m_nPixelColorFilter;
	}

	// Check whether the device supports BalanceRatioSelector
	emStatus = GXIsImplemented(m_hDevice, GX_ENUM_BALANCE_RATIO_SELECTOR, &m_bBalanceRatioSelector);
	VERIFY_STATUS_RET(emStatus);

	// Get image height
	emStatus = GXGetInt(m_hDevice, GX_INT_HEIGHT, &m_nImageHeight);
	m_stFlatFieldCorrection.nImgHei = (VxUint32)m_nImageHeight;

	// Get image actual bits
	m_stFlatFieldCorrection.nActualBits = m_nActualBits;

	return emStatus;
}

//----------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to close device

\return  void
*/
//----------------------------------------------------------------------------------
void CGxFlatFieldCorrectionDlg::OnBnClickedBtnCloseDevice()
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	// If the device is acquiring image then stop it
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

		m_bIsSnap                = false;
		m_bFFCGray               = false;

		m_bExecuteFFC            = false;
		m_bGetFFCCoefficients    = false;
		m_bIsFFC                 = false;
		m_bBrightImgAcquired     = false;
		m_bBrightImgAcquiredOnce = false;
		m_bDarkImgAcquired       = false;
		m_bDarkImgAcquiredOnce   = false;

	}

	emStatus = GXCloseDevice(m_hDevice);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		// TODO: Add your error handler code here 
	}

	m_bDevOpened             = false;
	m_hDevice                = NULL;

	// Release memory for showing image
	UnPrepareForShowImg();

	KillTimer(0);

	UpDateUI();
	GetDlgItem(IDC_STATIC_PROMPT_INFOR)->SetWindowText("");
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to stop acquisition

\return  void
*/
//----------------------------------------------------------------------------------
void CGxFlatFieldCorrectionDlg::OnBnClickedBtnStopSnap()
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	//Send AcquisitionStop command 
	emStatus = GXSendCommand(m_hDevice, GX_COMMAND_ACQUISITION_STOP);
	GX_VERIFY(emStatus);

	//Unregister the CaptureCallback function
	emStatus = GXUnregisterCaptureCallback(m_hDevice);
	GX_VERIFY(emStatus);

	m_bExecuteFFC            = false;
	m_bGetFFCCoefficients    = false;
	m_bIsFFC                 = false;
	m_bBrightImgAcquired     = false;
	m_bBrightImgAcquiredOnce = false;
	m_bDarkImgAcquired       = false;
	m_bDarkImgAcquiredOnce   = false;
	m_bIsSnap                = false;

	UpDateUI();
	GetDlgItem(IDC_STATIC_PROMPT_INFOR)->SetWindowText("");
}

//----------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to start acquisition

\return  void
*/
//----------------------------------------------------------------------------------
void CGxFlatFieldCorrectionDlg::OnBnClickedBtnStartSnap()
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_ERROR;

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

	m_bIsSnap = true;

	UpDateUI();

	GetDlgItem(IDC_STATIC_PROMPT_INFOR)->SetWindowText("1. Performing flat field correction needs acquire dark field and bright field images.\r\n2. Color camera is strongly recommended to perform white balance first.\r\n3. Cover the lens cover when acquiring dark field images.");
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize the Gain control
 
\return  void
*/
//----------------------------------------------------------------------------------
void CGxFlatFieldCorrectionDlg::InitGainUI()
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
void CGxFlatFieldCorrectionDlg::InitShutterUI()
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
\brief   This function demonstrates how to initialize the Gray control

\return  void
*/
//----------------------------------------------------------------------------------
void CGxFlatFieldCorrectionDlg::InitGrayUI()
{
	CSliderCtrl     *pSliderCtrl  = (CSliderCtrl *)GetDlgItem(IDC_SLIDER_GRAY);
	CEdit           *pEditGrayVal = (CEdit *)GetDlgItem(IDC_EDIT_GRAY);
	CSpinButtonCtrl *pSpin        = (CSpinButtonCtrl*)GetDlgItem(IDC_SPIN_EXPECT_GRAY_VALUE);

	// Check whether the control is available
	if ((pSliderCtrl == NULL) || (pEditGrayVal == NULL) || (pSpin == NULL) )
	{
		return;
	}

	// Set the range and current value of Gray control
	pSliderCtrl->SetRange(GRAY_MIN, GRAY_MAX, false);
	pSliderCtrl->SetPos(GRAY_INIT);
	pSpin->SetRange(GRAY_MIN, GRAY_MAX);
	m_nNewGray = GRAY_INIT;

	UpdateData(FALSE);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize the FrameCount control

\return  void
*/
//----------------------------------------------------------------------------------
void CGxFlatFieldCorrectionDlg::InitFrameCountUI()
{
	CSliderCtrl *pSliderCtrl  = (CSliderCtrl *)GetDlgItem(IDC_SLIDER_FRAME_NUM);

	// Check whether the control is available
	if (NULL == pSliderCtrl)
	{
		return;
	}

	// Set the range and current value of FrameCount control
	pSliderCtrl->SetRange(CORRECTION_UI_FRAME_COUNT_MIN, CORRECTION_UI_FRAME_COUNT_MAX, true);
	pSliderCtrl->SetPos(CORRECTION_UI_FRAME_COUNT_2);

	UpdateData(FALSE);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to set the value of BalanceRatioSelector

\return  void
*/
//----------------------------------------------------------------------------------
void CGxFlatFieldCorrectionDlg::OnCbnSelchangeComboRatioSelector()
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
void CGxFlatFieldCorrectionDlg::OnEnKillfocusEditBalanceRatio()
{
	// TODO: Add your control notification handler code here
	// Judge validity of handle, avoid exception due to closing program after device off-line
	if (m_hDevice == NULL)
	{
		return;
	}

	UpdateData(TRUE);
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	// If greater than Max then set BalanceRatio as Max
	if (m_dBalanceRatio > m_stFloatRange.dMax)
	{
		m_dBalanceRatio = m_stFloatRange.dMax;
	}
	
	// If less than Min then set BalanceRatio as Min
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

\return  Success: TRUE   Fail:FALSE
*/
//-----------------------------------------------------------------------
BOOL CGxFlatFieldCorrectionDlg::PreTranslateMessage(MSG* pMsg)
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
		case IDC_EDIT_GRAY: 

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
void CGxFlatFieldCorrectionDlg::OnKillfocusEditShutter() 
{
	// TODO: Add your control notification handler code here
	// Judge validity of handle, avoid exception due to closing program after device off-line
	if (m_hDevice == NULL)
	{
		return;
	}

	UpdateData();	
	GX_STATUS status = GX_STATUS_SUCCESS;

	// If greater than Max then set ExposureTime as Max
	if (m_dShutterValue > m_stShutterFloatRange.dMax)
	{
		m_dShutterValue = m_stShutterFloatRange.dMax;
	}
	// If less than Min then set ExposureTime as Min
	if (m_dShutterValue < m_stShutterFloatRange.dMin)
	{
		m_dShutterValue = m_stShutterFloatRange.dMin;
	}
	
	status = GXSetFloat(m_hDevice, GX_FLOAT_EXPOSURE_TIME, m_dShutterValue);
	GX_VERIFY(status);

	UpdateData(FALSE);
}

//--------------------------------------------------------------------
/**
\brief   This function demonstrates how to set the Gain

\return  void
*/
//--------------------------------------------------------------------
void CGxFlatFieldCorrectionDlg::OnKillfocusEditGain()
{
	// TODO: Add your control notification handler code here
	// Judge validity of handle, avoid exception due to closing program after device off-line
	if (m_hDevice == NULL)
	{
		return;
	}
	
	UpdateData();
	GX_STATUS status = GX_STATUS_SUCCESS;

	// If greater than Max then set Gain as Max
	if (m_dGainValue > m_stGainFloatRange.dMax)
	{
		m_dGainValue = m_stGainFloatRange.dMax;
	}
	
	// If less than Min then set Gain as Min
	if (m_dGainValue < m_stGainFloatRange.dMin)
	{
		m_dGainValue = m_stGainFloatRange.dMin;
	}
	
	status = GXSetFloat(m_hDevice, GX_FLOAT_GAIN, m_dGainValue);
	GX_VERIFY(status);
	
	UpdateData(FALSE);
}

//--------------------------------------------------------------------
/**
\brief   This function demonstrates how to set the FFC expected gray value

\return  void
*/
//--------------------------------------------------------------------
void CGxFlatFieldCorrectionDlg::OnKillfocusEditGray()
{
	// TODO: Add your control notification handler code here
	// Judge validity of handle, avoid exception due to closing program after device off-line
	if (m_hDevice == NULL)
	{
		return;
	}

	UpdateData();
	GX_STATUS status = GX_STATUS_SUCCESS;

	// If greater than Max then set Gray as Max
	if (m_nNewGray > GRAY_MAX)
	{
		m_nNewGray = GRAY_MAX;
	}

	// If less than Min then set Gray as Min
	if (m_nNewGray < GRAY_MIN)
	{
		m_nNewGray = GRAY_MIN;
	}
	m_sliderGrayValue.SetPos(m_nNewGray);

	UpdateData(FALSE);
}

//----------------------------------------------------------------
/**
\brief  This timer callback function demonstrates how to update the value of BalanceWhiteAuto to UI
\param  nIDEvent   [in]

\return void
*/
//----------------------------------------------------------------
void CGxFlatFieldCorrectionDlg::OnTimer(UINT_PTR nIDEvent) 
{
	// TODO: Add your message handler code here and/or call default
	GX_STATUS emStatus = GX_STATUS_ERROR;

	// Get the current value of BalanceWhiteAuto from the device and update to UI
	if (m_nAutoWhiteBalance == GX_BALANCE_WHITE_AUTO_ONCE)
	{
		int64_t nWBEnumVal = 0;   

		CComboBox *pAutoWBBox = (CComboBox *)GetDlgItem(IDC_COMBO_AWB);

		// Get the enumeration entry of BalanceWhiteAuto
		emStatus = GXGetEnum(m_hDevice, GX_ENUM_BALANCE_WHITE_AUTO, &nWBEnumVal);
		if (emStatus != GX_STATUS_SUCCESS)
		{
			return;
		}

		m_nAutoWhiteBalance = nWBEnumVal;

		// Check whether the BalanceWhiteAuto is OFF
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

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to use the sliders to adjust FrameCount and Gray
\param   nSBCode     
\param   nPos        
\param   pScrollBar  

\return  void
*/
//----------------------------------------------------------------------------------
void CGxFlatFieldCorrectionDlg::OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar)
{
	// TODO: Add your message handler code here and/or call default
	char PutText[10] = {0};
	int  nValue      = 0;

	if (pScrollBar == (CScrollBar*)&m_sliderFrameCount)            // The FrameCount slider
	{
		m_nFrameCount = m_sliderFrameCount.GetPos();
		GetFrameCount(m_nFrameCount);
	}
	else if (pScrollBar == (CScrollBar*)&m_sliderGrayValue)        // The Gray slider
	{
		m_nNewGray = m_sliderGrayValue.GetPos();
		sprintf_s(PutText,"%d", m_nNewGray);
		GetDlgItem(IDC_EDIT_GRAY)->SetWindowText(PutText);	
	}

	UpdateData(FALSE);

	CDialog::OnHScroll(nSBCode, nPos, pScrollBar);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to use the Spin control to adjust the gray value
\param   pNMHDR     
\param   pResult    

\return  void
*/
//----------------------------------------------------------------------------------
void CGxFlatFieldCorrectionDlg::OnDeltaposSpinExpectGrayValue(NMHDR *pNMHDR, LRESULT *pResult)
{
	LPNMUPDOWN pNMUpDown = reinterpret_cast<LPNMUPDOWN>(pNMHDR);
	*pResult = 0;

	UpdateData();

	if(pNMUpDown->iDelta == -1)    // If the value is -1, the downward arrow of Spin control is clicked
	{
		m_nNewGray--;

		// If greater than Max then set Gray as Max
		if (m_nNewGray < GRAY_MIN)
		{
			m_nNewGray = GRAY_MIN;
		}
		m_sliderGrayValue.SetPos(m_nNewGray);
	}

	else if(pNMUpDown->iDelta == 1)  // If the value is 1, the upward arrow of Spin control is clicked
	{
		m_nNewGray++;

		// If less than Min then set Gray as Min
		if (m_nNewGray > GRAY_MAX)
		{
			m_nNewGray = GRAY_MAX;
		}
		m_sliderGrayValue.SetPos(m_nNewGray);
	}

	UpdateData(false);
}

//----------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to acquire dark field image

\return  void
*/
//----------------------------------------------------------------------------------
void CGxFlatFieldCorrectionDlg::OnBnClickedButtonAcquireDarkImg()
{
	// TODO: Add your control notification handler code here
	m_bDarkImgAcquired = true;
	m_bExecuteFFC      = false;
	m_bIsFFC           = false;

	UpDateUI();	
}

//----------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to acquire bright field image

\return  void
*/
//----------------------------------------------------------------------------------
void CGxFlatFieldCorrectionDlg::OnBnClickedButtonAcquireBrightImg()
{
	// TODO: Add your control notification handler code here
	memset(m_pAccumulateImgBuf, 0, sizeof(VxUint32) * (size_t)(m_nImageWidth * m_nImageHeight));

	m_bBrightImgAcquired     = true;
	m_bBrightImgAcquiredOnce = false;
	m_bExecuteFFC            = false;
	m_bIsFFC                 = false; 

	UpDateUI();	
}

//----------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to get flat field correction coefficients

\return  void
*/
//----------------------------------------------------------------------------------
void CGxFlatFieldCorrectionDlg::OnBnClickedButtonExecuteFFC()
{
	// TODO: Add your control notification handler code here
	VxInt32  emStatus = DX_OK;

	// Check whether the dark field image has been acquired
	if (false == m_bDarkImgAcquiredOnce)
	{
		m_stFlatFieldCorrection.pDarkBuf = NULL;
	}
	else
	{
		m_stFlatFieldCorrection.pDarkBuf = m_pDarkBuf;
	}

	// Check whether to use the set gray value when performing flat field correction
	m_nFFClength = (int)(4 * m_nImageWidth * m_nImageHeight);
	if (true == m_bFFCGray)
	{
		m_nFFCGray = m_nNewGray;
		emStatus = DxGetFFCCoefficients(m_stFlatFieldCorrection, m_pFFCCoefficients, &m_nFFClength, &m_nFFCGray);
	}
	else
	{
		emStatus = DxGetFFCCoefficients(m_stFlatFieldCorrection, m_pFFCCoefficients, &m_nFFClength);
	}

	//Check whether the calculation of flat field correction coefficients are successful
  	if (DX_OK != emStatus)
  	{
		GetDlgItem(IDC_STATIC_PROMPT_INFOR)->SetWindowText("Fail to calculate flat field correction coefficients.");
		m_bExecuteFFC         = false;                
		m_bGetFFCCoefficients = false;
		m_bIsFFC              = false;                 
  	} 
  	else
  	{
  		GetDlgItem(IDC_STATIC_PROMPT_INFOR)->SetWindowText("FFC is executed successfully.");
		m_bExecuteFFC         = true;        // FFC is executed successfully
		m_bShowFFCOK          = false;
		m_bGetFFCCoefficients = true;        // The coefficients of FFC are calculated successfully     
		m_bIsFFC              = true;        // Apply flat field correction coefficients to the acquired image
  	}

	UpDateUI();	
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates whether apply flat field correction coefficients to the acquired image

\return void
*/
//----------------------------------------------------------------------------------
void CGxFlatFieldCorrectionDlg::OnBnClickedCheckPreview()
{
	// TODO: Add your control notification handler code here
	m_bIsFFC = !m_bIsFFC;
	if (m_bIsFFC)
	{
		m_bShowFFCOK = false;
	}

	GetDlgItem(IDC_STATIC_PROMPT_INFOR)->SetWindowText(" ");
}

//-------------------------------------------------------------------
/**
\brief   This function demonstrates how to calculate the accumulated bright field image
\param   pImgBuf           [in]    Image in
\param   nDataWidth        [in]    Image bits
\param   nlength           [in]    Image size
\param   pAccumulateImgBuf [in]    Accumulated bright field image

\return  Status code
*/
//-------------------------------------------------------------------
DX_STATUS CGxFlatFieldCorrectionDlg::AccumulateImage(void *pImgBuf, int nDataWidth, int64_t nlength, VxUint32 *pAccumulateImgBuf)
{
	if (NULL == pImgBuf || NULL == pAccumulateImgBuf)
	{
		return DX_PARAMETER_INVALID;
	}

	// Check whether the image bits are valid
	if ((nDataWidth < IMAGE_DATA_NUM_MIN) || (nDataWidth > IMAGE_DATA_NUM_MAX))
	{
		return DX_PARAMETER_OUT_OF_BOUND;
	}

	if (nlength < 1)
	{
		return DX_PARAMETER_OUT_OF_BOUND;
	}

	// Calculate the accumulated bright field image
	if (IMAGE_DATA_NUM_MIN == nDataWidth)
	{

		unsigned char * pImgBufTemp = (unsigned char *)pImgBuf;
		for (int64_t i = 0; i < nlength; i++)
		{
			pAccumulateImgBuf[i] += pImgBufTemp[i];
		}

	}
	else
	{
		unsigned short * pImgBufTemp = (unsigned short *)pImgBuf;
		for (int64_t i = 0; i < nlength; i++)
		{
			pAccumulateImgBuf[i] += pImgBufTemp[i];
		}
	}

	return DX_OK;
}

//-------------------------------------------------------------------
/**
\brief  This function demonstrates how to calculate the average bright field image
\param  pImgBuf                [in]     Image in
\param  nlength                [in]     Image size
\param  nAccumulateNum         [in]     Accumulated number of images
\param  nDataWidth             [in]     Image bits
\param  pAverageImgBuf         [in]     Average bright field image

\return Status code
*/
//-------------------------------------------------------------------
DX_STATUS CGxFlatFieldCorrectionDlg::GetAverageImage(VxUint32 *pImgBuf, int64_t nlength, int nAccumulateNum, int nDataWidth, void *pAverageImgBuf)
{
	if (NULL == pImgBuf || NULL == pAverageImgBuf)
	{
		return DX_PARAMETER_INVALID;
	}

	// Check whether the image bits are valid
	if ((nDataWidth < IMAGE_DATA_NUM_MIN) || (nDataWidth > IMAGE_DATA_NUM_MAX))
	{
		return DX_PARAMETER_OUT_OF_BOUND;
	}

	if (nlength < 1 || nAccumulateNum < 1)
	{
		return DX_PARAMETER_OUT_OF_BOUND;
	}

	// Calculate the average bright field image
	int64_t i = 0;
	if (IMAGE_DATA_NUM_MIN == nDataWidth)
	{

		unsigned char *pAverageImgBufTemp = (unsigned char *)pAverageImgBuf;
		for (int64_t i = 0; i < nlength; i++)
		{
			pAverageImgBufTemp[i] = pImgBuf[i] / nAccumulateNum;
		}

	}
	else
	{
		unsigned short *pAverageImgBufTemp = (unsigned short *)pAverageImgBuf;
		for (int64_t i = 0; i < nlength; i++)
		{
			pAverageImgBufTemp[i] = pImgBuf[i]  / nAccumulateNum;
		}
	}

	return DX_OK;
}

//----------------------------------------------------------------------------------
/**
\brief   This function demonstrates whether to use the set gray value when performing flat field correction

\return  void
*/
//----------------------------------------------------------------------------------
void CGxFlatFieldCorrectionDlg::OnBnClickedCheckGray()
{
	// TODO: Add your control notification handler code here
	m_bFFCGray = !m_bFFCGray;
	UpDateUI();
}

//-------------------------------------------------------------------
/**
\brief   This function demonstrates how to calculate the average gray value of the acquired image
\param   pImgBuf   [in]     Image in         
\param   nImgSize  [in]     Image size

\return  GRAY_MAX or the average gray value
*/
//-------------------------------------------------------------------
int64_t CGxFlatFieldCorrectionDlg::GetAverageGray8bit(BYTE *pImgBuf, int64_t nImgSize)
{
	int64_t nGrayAverage = 0;
	int64_t nGraySum     = 0;

	if ((NULL == pImgBuf) || (nImgSize < 1))
	{
		return GRAY_INVALID;
	}

	// Calculate the average gray value
	for(int64_t i = 0; i < nImgSize; i++)
	{
		nGraySum += (int64_t) *(pImgBuf + i);
	}

	nGrayAverage = nGraySum / nImgSize;
	return nGrayAverage;
}

//-------------------------------------------------------------------
/**
\brief   This function demonstrates how to convert the set frame count to actual acquisition frame count
\param   nFrameCount   [in]     Acquisition frame count 

\return  void
*/
//-------------------------------------------------------------------
void CGxFlatFieldCorrectionDlg::GetFrameCount(int &nFrameCount)
{
	// Convert the set frame count to actual acquisition frame count
	switch (nFrameCount)   
	{
	case CORRECTION_UI_FRAME_COUNT_MIN:
		nFrameCount = CORRECTION_ACTUAL_FRAME_COUNT_1;
		break;

	case CORRECTION_UI_FRAME_COUNT_1:
		nFrameCount = CORRECTION_ACTUAL_FRAME_COUNT_2;
		break;

	case CORRECTION_UI_FRAME_COUNT_2:
		nFrameCount = CORRECTION_ACTUAL_FRAME_COUNT_4;
		break;

	case CORRECTION_UI_FRAME_COUNT_3:
		nFrameCount = CORRECTION_ACTUAL_FRAME_COUNT_8;
		break;

	case CORRECTION_UI_FRAME_COUNT_MAX:
		nFrameCount = CORRECTION_ACTUAL_FRAME_COUNT_16;
		break;

	default: 
		break;
	}
}

//-------------------------------------------------------------------
/**
\brief   This function demonstrates how to read flat field correction coefficients from the file
\param   strFilePath   [in]      The file path

\return  File error code
*/
//-------------------------------------------------------------------
FFC_STATUS CGxFlatFieldCorrectionDlg::ReadFFCCoefficientsFile(const std::string &strFilePath)
{
	FFC_STATUS emStatus = FFC_STATUS_SUCCESS;
	FILE *fp = NULL;

	do 
	{
		if (NULL == m_pFFCCoefficients)
		{
			emStatus = FFC_STATUS_FILE_DATA_ERROR;
			break;
		}

		// Check whether the file path is valid
		size_t nSuffix = strFilePath.rfind(".ffc");
		if (nSuffix != strFilePath.size() - strlen(".ffc"))
		{
			emStatus = FFC_STATUS_FILE_PATH_ERROR;
			break;
		}

		// Open the file suffixed with .ffc
		size_t nError = fopen_s(&fp, strFilePath.c_str(), "rb");
		if (0 != nError)
		{
			emStatus = FFC_STATUS_FILE_OPEN_ERROR;
			break;
		}

		//Check the flat field correction(FFC) coefficients length(byte)
		fseek(fp, 0L, SEEK_END);
		size_t nFileLength = ftell(fp);
		if (((2 * m_nImageWidth * m_nImageHeight) != nFileLength) && ((4 * m_nImageWidth * m_nImageHeight) != nFileLength))
		{
			emStatus = FFC_STATUS_FILE_READ_ERROR;
			break;
		}
		else
		{	
			m_nFFClength = (int)nFileLength;
		}

		// Read flat field correction coefficients from the file
		fseek(fp, 0, SEEK_SET);
		nError = fread(m_pFFCCoefficients, m_nFFClength, 1, fp);
		if (1 != nError)
		{
			emStatus = FFC_STATUS_FILE_READ_ERROR;
			break;
		}
	} while (0);

	if (fp)
	{
		fclose(fp);
	}

	return emStatus;
}

//-------------------------------------------------------------------
/**
\brief   This function demonstrates how to write flat field correction coefficients to the file
\param   strFilePath   [in]    The file path   

\return  File error code
*/
//-------------------------------------------------------------------
FFC_STATUS CGxFlatFieldCorrectionDlg::WriteFFCCoefficientsFile(const std::string &strFilePath)
{
	FFC_STATUS emStatus = FFC_STATUS_SUCCESS;
	FILE       *fp      = NULL;

 	do 
 	{
 		if (NULL == m_pFFCCoefficients)
 		{
 			emStatus = FFC_STATUS_FILE_DATA_ERROR;
 			break;
 		}
 
		// Check whether the file path is valid
		size_t nSuffix = strFilePath.rfind(".ffc");
		if (nSuffix != strFilePath.size() - strlen(".ffc"))
		{
			emStatus = FFC_STATUS_FILE_PATH_ERROR;
			break;
		}

		// Create a file suffixed with .ffc
		int nError = fopen_s(&fp, strFilePath.c_str(), "wb");
		if (0 != nError)
		{
			emStatus = FFC_STATUS_FILE_OPEN_ERROR;
			break;
		}

		// Write flat field correction coefficients to the file suffixed with .ffc
		if (1 != fwrite(m_pFFCCoefficients, m_nFFClength, 1, fp))
		{
			emStatus = FFC_STATUS_FILE_WRITE_ERROR;
			break;
		}

	} while (0);

	if (fp)
	{
		fclose(fp);
	}

	return emStatus;
}

//----------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to load flat field correction coefficients

\return  void
*/
//----------------------------------------------------------------------------------
void CGxFlatFieldCorrectionDlg::OnBnClickedButtonRead()
{
	// TODO: Add your control notification handler code here
	FFC_STATUS emStatus = FFC_STATUS_SUCCESS;
	CString    strFilePath = "";

	// Load flat field correction coefficients from the file suffixed with .ffc
	// Apply flat field correction coefficients if loading is successful
	CFileDialog objFileDialog( TRUE, NULL, NULL, 0, "ffc file (*.ffc)|*.ffc||", this);
	if( objFileDialog.DoModal() == IDOK )
	{
		strFilePath = objFileDialog.GetPathName();	           // Get the file path		
		emStatus = ReadFFCCoefficientsFile(strFilePath.GetBuffer(0));
		switch (emStatus)   
		{
		case FFC_STATUS_FILE_DATA_ERROR:
			MessageBox("Invalid parameter!");
			break;

		case FFC_STATUS_FILE_PATH_ERROR:
			MessageBox("Invalid file path!");
			break;

		case FFC_STATUS_FILE_OPEN_ERROR:
			MessageBox("Fail to open the file!");
			break;

		case FFC_STATUS_FILE_READ_ERROR:
			MessageBox("Fail to load the parameters!");
			break;

		default: 
			break;
		}

		if (FFC_STATUS_SUCCESS == emStatus)
		{
			m_bExecuteFFC         = true;
			m_bShowFFCOK          = false;
			m_bIsFFC              = true;
			m_bGetFFCCoefficients = true;
		}
		else
		{
			m_bExecuteFFC         = false;
			m_bIsFFC              = false;
			m_bGetFFCCoefficients = false;
			UpDateUI();
			GetDlgItem(IDC_STATIC_PROMPT_INFOR)->SetWindowText(" ");
		}
	}	
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to save flat field correction coefficients

\return void
*/
//----------------------------------------------------------------------------------
void CGxFlatFieldCorrectionDlg::OnBnClickedButtonWrite()
{
	// TODO: Add your control notification handler code here
	FFC_STATUS emStatus    = FFC_STATUS_SUCCESS;
	CString    strFilePath = "";

	if (false == m_bGetFFCCoefficients)
	{
		MessageBox("Invalid parameter!");
		return ;
	}

	// Save the FFC coefficients in binary form to the file suffixed with .ffc
	CFileDialog objFileDialog( FALSE, ".ffc", "FlatFieldCorrection.ffc", OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, "ffc file (*.ffc)|*.ffc||", this);
	if( objFileDialog.DoModal() == IDOK )
	{
		strFilePath = objFileDialog.GetPathName();	           // Get the file path	
		emStatus    = WriteFFCCoefficientsFile(strFilePath.GetBuffer(0));
		switch (emStatus)   
		{
		case FFC_STATUS_FILE_DATA_ERROR:
			MessageBox("Invalid parameter!");
			break;

		case FFC_STATUS_FILE_PATH_ERROR:
			MessageBox("Invalid file path!");
			break;

		case FFC_STATUS_FILE_OPEN_ERROR:
			MessageBox("Fail to open the file!");
			break;

		case FFC_STATUS_FILE_WRITE_ERROR:
			MessageBox("Fail to write the parameters!");
			break;

		default: 
			break;
		}  //end of switch (emStatus)
	}	
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to refresh white ratio 

\return  void
*/
//---------------------------------------------------------------------------------
void CGxFlatFieldCorrectionDlg::RefreshWhiteRatio()
{
	GX_STATUS emStatus = GX_STATUS_ERROR;
	double    dValue   = 0.0;
	CString   strTemp  = _T("");

	// Get current value of white ratio and update to the UI
	emStatus = GXGetFloat(m_hDevice, GX_FLOAT_BALANCE_RATIO, &dValue);
	GX_VERIFY(emStatus);

	m_dBalanceRatio = dValue;
	strTemp.Format("%.4f",dValue);
	SetDlgItemText(IDC_EDIT_BALANCE_RATIO, strTemp);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to acquire dark field image

\return  void
*/
//---------------------------------------------------------------------------------
void CGxFlatFieldCorrectionDlg::AcquireDarkFieldImg(void)
{
	// Calculate the average gray value of the dark field image
	m_nAverageGrayDarkImg = (int)GetAverageGray8bit(m_pDarkBuf, m_nImageWidth * m_nImageHeight);
	if (GRAY_INVALID == m_nAverageGrayDarkImg)
	{
		m_bDarkImgAcquiredOnce = false;
		MessageBox("Acquisition of dark field image fails, please reacquire!");
	}
	else if (m_nAverageGrayDarkImg > BRIGHT_IMG_GRAY_MIN)
	{
		m_bDarkImgAcquiredOnce = false;
		GetDlgItem(IDC_STATIC_PROMPT_INFOR)->SetWindowText("The dark field image is too bright, the flat field correction will not use the dark field image, and this will not affect the effect of flat field correction.");
	} 
	else
	{
		m_bDarkImgAcquiredOnce = true;
		GetDlgItem(IDC_STATIC_PROMPT_INFOR)->SetWindowText("Acquisition of dark field image completes.");
	}
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to acquire bright field image

\return  void
*/
//---------------------------------------------------------------------------------
void CGxFlatFieldCorrectionDlg::AcquireBrightFieldImg(void)
{
	DX_STATUS emStatus = DX_OK;
	int64_t   nImgSize = 0;
	nImgSize = m_nImageWidth * m_nImageHeight;

	// Acquire bright field image
	do 
	{
		// Get accumulated bright field image
		emStatus = AccumulateImage(m_pBrightBuf, m_nActualBits, nImgSize, m_pAccumulateImgBuf);
		if (DX_OK != emStatus)
		{
			m_nBrightImgNum = 0;
			m_bBrightImgAcquired = false;
			MessageBox("The accumulated bright field image is invalid, please reacquire the bright field image!");
			break;
		}

		m_nBrightImgNum++;
		if (m_nFrameCount == m_nBrightImgNum)
		{
			// Get average bright field image
			emStatus = GetAverageImage(m_pAccumulateImgBuf, nImgSize, m_nBrightImgNum, m_nActualBits, m_pAverageImgBuf);
			if (DX_OK != emStatus)
			{
				m_nBrightImgNum = 0;
				m_bBrightImgAcquired = false;
				MessageBox("The average bright field image is invalid, please reacquire the bright field image!");
				break;
			}

			// Calculate the average gray value of the bright field image
			m_nAverageGrayBrightImg = (int)GetAverageGray8bit(m_pAverageImgBuf, nImgSize);
			if (GRAY_INVALID == m_nAverageGrayBrightImg)
			{
				MessageBox("Acquisition of bright field image fails, please reacquire!");
			}
			else if (m_nAverageGrayBrightImg < BRIGHT_IMG_GRAY_MIN)
			{
				GetDlgItem(IDC_STATIC_PROMPT_INFOR)->SetWindowText("The bright field image is too dark, it will affect the flat field correction effect. It is recommended to adjust the image brightness in the range of 20-250.");
				m_bBrightImgAcquiredOnce = true;
				UpDateUI();
			} 
			else if (m_nAverageGrayBrightImg > BRIGHT_IMG_GRAY_MAX)
			{
				GetDlgItem(IDC_STATIC_PROMPT_INFOR)->SetWindowText("The bright field image is too bright, it will affect the flat field correction effect. It is recommended to adjust the image brightness in the range of 20-250.");
				m_bBrightImgAcquiredOnce = true;
				UpDateUI();
			} 
			else
			{
				GetDlgItem(IDC_STATIC_PROMPT_INFOR)->SetWindowText("Acquisition of bright field images completes.");
				m_bBrightImgAcquiredOnce = true;
				UpDateUI();
			}

			m_nBrightImgNum = 0;
			m_bBrightImgAcquired = false;
		}
	} while (0);
}
