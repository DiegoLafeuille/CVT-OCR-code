// GxSingleCamDlg.cpp : Defines the class behaviors for the application.
//

#include "stdafx.h"
#include "GxSingleCam.h"
#include "GxSingleCamDlg.h"
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

// Implementation
protected:
	DECLARE_MESSAGE_MAP()
public:
	CString m_strStaticProductVersion;
	CString m_strStaticLegalCopyright;
};

CAboutDlg::CAboutDlg() : CDialog(CAboutDlg::IDD)
, m_strStaticProductVersion(_T(""))
, m_strStaticLegalCopyright(_T(""))
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_STATIC_PRODUCTVERSION, m_strStaticProductVersion);
	DDX_Text(pDX, IDC_STATIC_LEGALCOPYRIGHT, m_strStaticLegalCopyright);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialog)
END_MESSAGE_MAP()


// CGxSingleCamDlg Dialog
CGxSingleCamDlg::CGxSingleCamDlg(CWnd* pParent /*=NULL*/)
	: CDialog(CGxSingleCamDlg::IDD, pParent)
	, m_bIsOpen(false)
	, m_bIsSnap(false)
	, m_bColorFilter(false)
	, m_bTriggerMode(false)
	, m_bTriggerSource(false)
	, m_bTriggerActive(false)
	, m_bBalanceWhiteAuto(false)
	, m_bBalanceWhiteRatioSelect(false)
	, m_strSavePath("")
	, m_strBalanceWhiteAutoMode("Off")
	, m_pWnd(NULL)
    , m_pSampleCaptureEventHandle(NULL)
	, m_pBitmap(NULL)
	, m_bCheckSaveBmp(FALSE)
	, m_dEditShutterValue(0)
	, m_dEditGainValue(0)
	, m_dEditBalanceRatioValue(0)
	, m_nTriggerModeOld(0)
	, m_nTriggerSourceOld(0)
	, m_nTriggerActiveOld(0)
	, m_nBalanceWhiteAutoOld(0)
	, m_nBanlanceWhiteRatioOld(0)
	, m_dShutterValueMax(0)
	, m_dShutterValueMin(0)
	, m_dGainValueMax(0)
	, m_dGainValueMin(0)
	, m_dBalanceWhiteRatioMax(0)
	, m_dBalanceWhiteRatioMin(0)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CGxSingleCamDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_Check(pDX, IDC_CHECK_SAVE_BMP, m_bCheckSaveBmp);
	DDX_Text(pDX, IDC_EDIT_SHUTTER, m_dEditShutterValue);
	DDX_Text(pDX, IDC_EDIT_GAIN, m_dEditGainValue);
	DDX_Text(pDX, IDC_EDIT_BALANCE_WHITE_SERISE, m_dEditBalanceRatioValue);
}

BEGIN_MESSAGE_MAP(CGxSingleCamDlg, CDialog)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	//}}AFX_MSG_MAP
	ON_BN_CLICKED(IDC_BTN_OPEN_DEVICE, &CGxSingleCamDlg::OnBnClickedBtnOpenDevice)
	ON_BN_CLICKED(IDC_BTN_CLOSE_DEVICE, &CGxSingleCamDlg::OnBnClickedBtnCloseDevice)
	ON_BN_CLICKED(IDC_BTN_START_SNAP, &CGxSingleCamDlg::OnBnClickedBtnStartSnap)
	ON_BN_CLICKED(IDC_BTN_STOP_SNAP, &CGxSingleCamDlg::OnBnClickedBtnStopSnap)
	ON_CBN_SELCHANGE(IDC_COMBO_TRIGGER_MODE, &CGxSingleCamDlg::OnCbnSelchangeComboTriggerMode)
	ON_CBN_SELCHANGE(IDC_COMBO_TRIGGER_SOURCE, &CGxSingleCamDlg::OnCbnSelchangeComboTriggerSource)
	ON_CBN_SELCHANGE(IDC_COMBO_TRIGGER_ACTIVE, &CGxSingleCamDlg::OnCbnSelchangeComboTriggerActive)
	ON_BN_CLICKED(IDC_CHECK_SAVE_BMP, &CGxSingleCamDlg::OnBnClickedCheckSaveBmp)
	ON_WM_CLOSE()
	ON_CBN_SELCHANGE(IDC_COMBO_BALANCE_WHITE_AUTO, &CGxSingleCamDlg::OnCbnSelchangeComboBalanceWhiteAuto)
	ON_CBN_SELCHANGE(IDC_COMBO_BALANCE_WHITE_SELECT, &CGxSingleCamDlg::OnCbnSelchangeComboBalanceWhiteSelect)
	ON_BN_CLICKED(IDC_BTN_SOFTTRIGGER, &CGxSingleCamDlg::OnBnClickedBtnSofttrigger)
	ON_EN_KILLFOCUS(IDC_EDIT_GAIN, &CGxSingleCamDlg::OnEnKillfocusEditGain)
	ON_EN_KILLFOCUS(IDC_EDIT_BALANCE_WHITE_SERISE, &CGxSingleCamDlg::OnEnKillfocusEditBalanceWhiteSerise)
	ON_WM_TIMER()
	ON_EN_KILLFOCUS(IDC_EDIT_SHUTTER, &CGxSingleCamDlg::OnEnKillfocusEditShutter)

END_MESSAGE_MAP()

// CGxSingleCamDlg message handlers
BOOL CGxSingleCamDlg::OnInitDialog()
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

		//Initialize the window pointer with the picture widget
		m_pWnd = GetDlgItem(IDC_SHOW_PICTURE_STATIC);
		m_pSampleCaptureEventHandle = new CSampleCaptureEventHandler();

		//Get the current path of app
		char    strFileName[MAX_PATH] = {0};
		string  strSavePath           = "";
		size_t  nPos                  = 0;

		GetModuleFileName(NULL,(LPCH)strFileName,MAX_PATH);
		strSavePath = strFileName;
		nPos = strSavePath.find_last_of('\\');
		m_strSavePath = strSavePath.substr(0,nPos);
		m_strSavePath = m_strSavePath +"\\GxSingleCamImages";

		//Setting a timer
		SetTimer(0,1000,NULL);

		//Update UI
		__UpdateUI();
	}
	catch (CGalaxyException& e)
	{
		if (m_pSampleCaptureEventHandle != NULL)
		{
			delete m_pSampleCaptureEventHandle;
			m_pSampleCaptureEventHandle = NULL;
		}

		MessageBox(e.what());
		return FALSE;
		
	}
	catch (std::exception& e)
	{
		if (m_pSampleCaptureEventHandle != NULL)
		{
			delete m_pSampleCaptureEventHandle;
			m_pSampleCaptureEventHandle = NULL;
		}

		MessageBox(e.what());
		return FALSE;

	}

	return TRUE;  // return TRUE  unless you set the focus to a control
}

void CGxSingleCamDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		char strFileName[MAX_PATH] = {'\0'};
		GetModuleFileName(NULL, strFileName, MAX_PATH);
		CFileVersion fv(strFileName);
		CAboutDlg dlgAbout;
		dlgAbout.m_strStaticProductVersion = _T("Version: ") + fv.GetProductVersion();
		dlgAbout.m_strStaticLegalCopyright = fv.GetLegalCopyright();
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

void CGxSingleCamDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Make the icon in the center of client rectangle
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
HCURSOR CGxSingleCamDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

void CGxSingleCamDlg::OnBnClickedBtnOpenDevice()
{
	// TODO: Add your control notification handler code here
	bool bIsDeviceOpen = false;         ///< The flag indicates whether the device has been opened.
	bool bIsStreamOpen = false;         ///< The flag indicates whether the stream has been opened.

	try
	{
		//Enumerate all camera devices
		GxIAPICPP::gxdeviceinfo_vector vectorDeviceInfo;
		IGXFactory::GetInstance().UpdateDeviceList(1000,vectorDeviceInfo);

	    //Check it has found device or not.If not, will be hited with a dialog
		if (vectorDeviceInfo.size() <= 0)
		{
			MessageBox("No device found!");
			return;
		}

		//Open device
		m_objDevicePtr = IGXFactory::GetInstance().OpenDeviceBySN(vectorDeviceInfo[0].GetSN(),GX_ACCESS_EXCLUSIVE);
		bIsDeviceOpen = true;
		m_objFeatureControlPtr = m_objDevicePtr->GetRemoteFeatureControl();

		//Check whether the pointer of drawing image is null or not
		if (m_pBitmap != NULL)
		{
			delete m_pBitmap;
			m_pBitmap = NULL;
		}

		// Create memory for the object of drawing image
		m_pBitmap = new CGXBitmap(m_objDevicePtr,m_pWnd);

		//Check the device has stream or not.If more than zero, will be opend the stream.
		int nStreamCount = m_objDevicePtr->GetStreamCount();

		if (nStreamCount > 0)
		{
			m_objStreamPtr = m_objDevicePtr->OpenStream(0);
			m_objStreamFeatureControlPtr = m_objStreamPtr->GetFeatureControl();
			bIsStreamOpen = true;
		}
		else
		{
			throw exception("The device has been opend doesn't has stream!");
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

		//initialize device param
		__InitParam();

		//Initialize UI
		__InitUI();
		m_bIsOpen = true;

		//Update UI
		__UpdateUI();
	}
	catch (CGalaxyException& e)
	{
		//Check whether the device stream has been opened or not
		if (bIsStreamOpen)
		{
			m_objStreamPtr->Close();
		}

		//Check whether the device has been opened or not
		if (bIsDeviceOpen)
		{
			m_objDevicePtr->Close();
		}

		if (m_pBitmap != NULL)
		{
			delete m_pBitmap;
			m_pBitmap = NULL;
		}

		MessageBox(e.what());
		return;
	}
	catch (std::exception& e)
	{
		//Check whether the device stream has been opened or not
		if (bIsStreamOpen)
		{
			m_objStreamPtr->Close();
		}

		//Check whether the device has been opened or not
		if (bIsDeviceOpen)
		{
			m_objDevicePtr->Close();
		}

		if (m_pBitmap != NULL)
		{
			delete m_pBitmap;
			m_pBitmap = NULL;
		}

		MessageBox(e.what());
		return;
	}
}

//---------------------------------------------------------------------------------
/**
\brief   initialize device param

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamDlg::__InitParam()
{
	bool bBalanceWhiteAutoRead = false;         ///< The flag indicated whether auto white balance is available to read or not.

	//Set the continuous frame acquisition mode
	m_objFeatureControlPtr->GetEnumFeature("AcquisitionMode")->SetValue("Continuous");

	//Check whether support selecting trigger mode
	m_bTriggerMode = m_objFeatureControlPtr->IsImplemented("TriggerMode");
	if (m_bTriggerMode)
	{
		//Set the TriggerMode off
		m_objFeatureControlPtr->GetEnumFeature("TriggerMode")->SetValue("Off");
	}

	//Check whether support color filter
	m_bColorFilter = m_objFeatureControlPtr->IsImplemented("PixelColorFilter");

	//Check whether support selecting trigger source
	m_bTriggerSource = m_objFeatureControlPtr->IsImplemented("TriggerSource");

	//Check whether support selecting trigger activation
	m_bTriggerActive = m_objFeatureControlPtr->IsImplemented("TriggerActivation");

	//Check whether support auto white balance
	m_bBalanceWhiteAuto = m_objFeatureControlPtr->IsImplemented("BalanceWhiteAuto");

	//Get the flag whether auto white balance is available to read or not.
	bBalanceWhiteAutoRead   = m_objFeatureControlPtr->IsReadable("BalanceWhiteAuto");

	//If support auto white balance and is readable, get the value of auto white balance
	if (m_bBalanceWhiteAuto)
	{
		if (bBalanceWhiteAutoRead)
		{
			m_strBalanceWhiteAutoMode = m_objFeatureControlPtr->GetEnumFeature("BalanceWhiteAuto")
				                                              ->GetValue();
		}
	}

	//Get the flag whether support white balance ratio selector or not.
	m_bBalanceWhiteRatioSelect = m_objFeatureControlPtr->IsImplemented("BalanceRatioSelector");

	//Get the max and min value for exposuer time , gain and white balance ratio
	m_dShutterValueMax      = m_objFeatureControlPtr->GetFloatFeature("ExposureTime")->GetMax();
	m_dShutterValueMin      = m_objFeatureControlPtr->GetFloatFeature("ExposureTime")->GetMin();
	m_dGainValueMax         = m_objFeatureControlPtr->GetFloatFeature("Gain")->GetMax();
	m_dGainValueMin         = m_objFeatureControlPtr->GetFloatFeature("Gain")->GetMin();
	m_dBalanceWhiteRatioMax = m_objFeatureControlPtr->GetFloatFeature("BalanceRatio")->GetMax();
	m_dBalanceWhiteRatioMin = m_objFeatureControlPtr->GetFloatFeature("BalanceRatio")->GetMin();
}

//---------------------------------------------------------------------------------
/**
\brief   Initialize UI

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamDlg::__InitUI()
{
	//Check whether the device support color filter
	if (m_bColorFilter)
	{
		//Initialize combox widget
		m_nTriggerModeOld        = __InitComBox("TriggerMode",
			                                    (CComboBox*)GetDlgItem(IDC_COMBO_TRIGGER_MODE),
			                                    m_bTriggerMode);
		m_nTriggerSourceOld      = __InitComBox("TriggerSource",
			                                    (CComboBox*)GetDlgItem(IDC_COMBO_TRIGGER_SOURCE),
			                                     m_bTriggerSource);
		m_nTriggerActiveOld      = __InitComBox("TriggerActivation",
			                                    (CComboBox*)GetDlgItem(IDC_COMBO_TRIGGER_ACTIVE),
			                                     m_bTriggerActive);
		m_nBanlanceWhiteRatioOld = __InitComBox("BalanceRatioSelector",
			                                     (CComboBox*)GetDlgItem(IDC_COMBO_BALANCE_WHITE_SELECT),
			                                     m_bBalanceWhiteRatioSelect);
		m_nBalanceWhiteAutoOld   = __InitComBox("BalanceWhiteAuto",
			                                    (CComboBox*)GetDlgItem(IDC_COMBO_BALANCE_WHITE_AUTO),
			                                     m_bBalanceWhiteAuto);
	}
	else
	{
		//Initialize combox widget
		m_nTriggerModeOld        = __InitComBox("TriggerMode",
			                                    (CComboBox*)GetDlgItem(IDC_COMBO_TRIGGER_MODE),
			                                    m_bTriggerMode);
		m_nTriggerSourceOld      = __InitComBox("TriggerSource",
			                                    (CComboBox*)GetDlgItem(IDC_COMBO_TRIGGER_SOURCE),
			                                    m_bTriggerSource);
		m_nTriggerActiveOld      = __InitComBox("TriggerActivation",
			                                    (CComboBox*)GetDlgItem(IDC_COMBO_TRIGGER_ACTIVE),
			                                    m_bTriggerActive);
	}
	
	//Initialize ExposureTime Widget
	__InitShutterTime();
	
	//Initialize Gain Widget
	__InitGain();

	//Initialize white balance ratio Widget
	__InitBalanceWhiteRatio();

}

//---------------------------------------------------------------------------------
/**
\brief   Initialize combox
\param   strName         [in]    function name
\param   pComboBox       [in]    the pointer of combox widget
\param   bIsImplemented  [in]    whether support the function on the emFeatureID variable behalf

\return  void
*/
//----------------------------------------------------------------------------------
int CGxSingleCamDlg::__InitComBox(gxstring strName, CComboBox* pComboBox, bool bIsImplemented)
{
	if ((pComboBox == NULL) || (!bIsImplemented))
	{
		return -1;
	}

	int     nCurSel = 0;
	string  strCurrentEnumList = "";                 // the current function of devoice
	GxIAPICPP::gxstring_vector vectorEnumEntryList;  // the enumeration item list
	CEnumFeaturePointer objEnumFeaturePtr = m_objFeatureControlPtr->GetEnumFeature(strName);

	//Clear all content of the combox widget
	pComboBox->ResetContent();

	//Get the current value of strName
	strCurrentEnumList = objEnumFeaturePtr->GetValue().c_str();

	//Get the enumeration item list of th strName
	vectorEnumEntryList = objEnumFeaturePtr->GetEnumEntryList();
	for (uint32_t i = 0; i<vectorEnumEntryList.size(); i++)
	{
		string strEnumList = vectorEnumEntryList[i].c_str();
		pComboBox->SetItemData(pComboBox->AddString(strEnumList.c_str()), i);
		if (strCurrentEnumList == vectorEnumEntryList[i].c_str())
		{
			nCurSel = i;
		}
	}

	pComboBox->SetCurSel(nCurSel);
	return nCurSel;
}

//---------------------------------------------------------------------------------
/**
\brief   Initialize ExposureTime Widget

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamDlg::__InitShutterTime()
{
	CStatic* pStcShutterShow    = (CStatic*)GetDlgItem(IDC_STATIC_SHUTTER_TIME);
	CEdit*   pEditShutterVal    = (CEdit*)GetDlgItem(IDC_EDIT_SHUTTER);

	//Check the pointer of widget is null or not
	if (pEditShutterVal == NULL || pStcShutterShow == NULL)
	{
		return;
	}

	CString   strShutterTime      = "";       ///< Current exposure time value
	gxstring  strShutterUnit      = "";       ///< exposure time unit

	//Get the current value of exposure time
	m_dEditShutterValue   = m_objFeatureControlPtr->GetFloatFeature("ExposureTime")->GetValue();

	//Get the unit of exposure time
    strShutterUnit = m_objFeatureControlPtr->GetFloatFeature("ExposureTime")->GetUnit();
	strShutterTime.Format("ExposureTime(%.4f~%.4f)%s",m_dShutterValueMin,m_dShutterValueMax,strShutterUnit.c_str());
	pStcShutterShow->SetWindowText(strShutterTime);
	
	UpdateData(FALSE);
}

//---------------------------------------------------------------------------------
/**
\brief   Initialize Gain Widget

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamDlg::__InitGain()
{
	CStatic* pStcGainShow    = (CStatic*)GetDlgItem(IDC_STATIC_GAIN);
	CEdit*   pEditGainVal    = (CEdit*)GetDlgItem(IDC_EDIT_GAIN);

	//Check the pointer of widget is null or not
	if (pEditGainVal == NULL || pStcGainShow == NULL)
	{
		return;
	}

	CString  strGain      = "";     ///< Current gain value
	string strGainUnit    = "";     ///< the unit of gain

	//Get the current value of gain
	m_dEditGainValue = m_objFeatureControlPtr->GetFloatFeature("Gain")->GetValue();
	
	//Get the unit of gain
	strGainUnit = m_objFeatureControlPtr->GetFloatFeature("Gain")->GetUnit().c_str();
	strGain.Format("Gain(%.4f~%.4f)%s",m_dGainValueMin,m_dGainValueMax,strGainUnit.c_str());
	pStcGainShow->SetWindowText(strGain);
	
	UpdateData(FALSE);

}

//---------------------------------------------------------------------------------
/**
\brief   Initialize white balance ratio Widget

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamDlg::__InitBalanceWhiteRatio()
{
	//Check whether the device support color filter
	if (!m_bColorFilter)
	{
		return;
	}

	CStatic* pStcBalanceRatioShow    = (CStatic*)GetDlgItem(IDC_STATIC_BALANCE_RATIO);
	CEdit*   pEditBalanceRatioVal    = (CEdit*)GetDlgItem(IDC_EDIT_BALANCE_WHITE_SERISE);

	//Check the pointer of widget is null or not
	if (pEditBalanceRatioVal == NULL || pStcBalanceRatioShow == NULL)
	{
		return;
	}

	CString strBalanceRatio      = "";        ///< Current white balance ratio value
	string  strBalanceRatioUnit  = "";        ///< The unit of white balance ratio

	//Get the value of white balance ratio
	m_dEditBalanceRatioValue = m_objFeatureControlPtr->GetFloatFeature("BalanceRatio")->GetValue();

	//Get the unit of white balance ratio
	strBalanceRatioUnit = m_objFeatureControlPtr->GetFloatFeature("BalanceRatio")->GetUnit().c_str();

	//Update the content of widget
	strBalanceRatio.Format("AutoWhiteBalanceRatio(%.4f~%.4f)%s",m_dBalanceWhiteRatioMin,
		                    m_dBalanceWhiteRatioMax,strBalanceRatioUnit.c_str());
    pStcBalanceRatioShow->SetWindowText(strBalanceRatio);
	
	UpdateData(FALSE);

}
//---------------------------------------------------------------------------------
/**
\brief   Update UI

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamDlg::__UpdateUI()
{
	GetDlgItem(IDC_BTN_OPEN_DEVICE)->EnableWindow(!m_bIsOpen);
	GetDlgItem(IDC_BTN_CLOSE_DEVICE)->EnableWindow(m_bIsOpen);
	GetDlgItem(IDC_BTN_START_SNAP)->EnableWindow(m_bIsOpen && !m_bIsSnap);
	GetDlgItem(IDC_BTN_STOP_SNAP)->EnableWindow(m_bIsOpen && m_bIsSnap);
	GetDlgItem(IDC_COMBO_TRIGGER_MODE)->EnableWindow(m_bIsOpen && m_bTriggerMode);
	GetDlgItem(IDC_COMBO_TRIGGER_SOURCE)->EnableWindow(m_bIsOpen && m_bTriggerSource);
	GetDlgItem(IDC_BTN_SOFTTRIGGER)->EnableWindow(m_bIsOpen);
	GetDlgItem(IDC_COMBO_TRIGGER_ACTIVE)->EnableWindow(m_bIsOpen && m_bTriggerActive);
	GetDlgItem(IDC_EDIT_SHUTTER)->EnableWindow(m_bIsOpen);
	GetDlgItem(IDC_EDIT_GAIN)->EnableWindow(m_bIsOpen);
	GetDlgItem(IDC_CHECK_SAVE_BMP)->EnableWindow(m_bIsOpen);
	GetDlgItem(IDC_COMBO_BALANCE_WHITE_AUTO)->EnableWindow(m_bIsOpen 
		                                                   && m_bBalanceWhiteAuto
														   && m_bColorFilter);

	GetDlgItem(IDC_COMBO_BALANCE_WHITE_SELECT)->EnableWindow(m_bIsOpen 
		                                                     && m_bBalanceWhiteRatioSelect
															 && m_bColorFilter);

	GetDlgItem(IDC_EDIT_BALANCE_WHITE_SERISE)->EnableWindow(m_bIsOpen
		                                                   && m_strBalanceWhiteAutoMode == "Off"
		                                                   && m_bBalanceWhiteRatioSelect
														   && m_bColorFilter);


}

void CGxSingleCamDlg::OnBnClickedBtnCloseDevice()
{
	// TODO: Add your control notification handler code here
	//lose focus
	SetFocus();

	try
	{
		//If the device is acquiring image then stop it.
		if (m_bIsSnap)
		{
			//Send AcquisitionStop command 
			m_objFeatureControlPtr->GetCommandFeature("AcquisitionStop")->Execute();

			//Stop stream capturing
			m_objStreamPtr->StopGrab();

			//Unregister the CaptureCallback function
			m_objStreamPtr->UnregisterCaptureCallback();
		}
	}
	catch(CGalaxyException)
	{
		//do noting
	}
    
	try
	{
		//Close stream
		m_objStreamPtr->Close();

	}
	catch(CGalaxyException)
	{
		//do noting
	}
	try
	{
		//Close device
		m_objDevicePtr->Close();
	}
	catch(CGalaxyException)
	{
		//do noting
	}

	

	m_bIsOpen = false;
	m_bIsSnap = false;

	//Update UI
	__UpdateUI();
	if (m_pBitmap != NULL)
	{
		delete m_pBitmap;
		m_pBitmap = NULL;
	}

}

void CGxSingleCamDlg::OnBnClickedBtnStartSnap()
{
	// TODO: Add your control notification handler code here
	try
	{
		//Set StreamBufferHandlingMode
		m_objStreamFeatureControlPtr->GetEnumFeature("StreamBufferHandlingMode")->SetValue("OldestFirst");

		//Register the CaptureCallback function
		m_objStreamPtr->RegisterCaptureCallback(m_pSampleCaptureEventHandle,this);

		//Start stream capturing
		m_objStreamPtr->StartGrab();

	   //Send AcquisitionStart command 
		m_objFeatureControlPtr->GetCommandFeature("AcquisitionStart")->Execute();
		m_bIsSnap = true;

		//Update UI
		__UpdateUI();
	}
	catch (CGalaxyException& e)
	{
		MessageBox(e.what());
		return;	
	}
	catch (std::exception& e)
	{
		MessageBox(e.what());
		return;	
	}
}

void CGxSingleCamDlg::OnBnClickedBtnStopSnap()
{
	// TODO: Add your control notification handler code here
	try
	{
		//Send AcquisitionStop command 
		m_objFeatureControlPtr->GetCommandFeature("AcquisitionStop")->Execute();

		//Stop stream capturing
		m_objStreamPtr->StopGrab();

		//Unregister the CaptureCallback function
		m_objStreamPtr->UnregisterCaptureCallback();
		m_bIsSnap = false;

		//Update UI
		__UpdateUI();
	}
	catch (CGalaxyException& e)
	{
		MessageBox(e.what());
		return;	
	}
	catch (std::exception& e)
	{
		MessageBox(e.what());
		return;	
	}
}

void CGxSingleCamDlg::OnBnClickedBtnSofttrigger()
{
	// TODO: Add your control notification handler code here
	try
	{
		//Send TriggerSoftware command(It is valid in the status of opening trigger mode)
		m_objFeatureControlPtr->GetCommandFeature("TriggerSoftware")->Execute();
	}
	catch (CGalaxyException& e)
	{
		MessageBox(e.what());
		return;	
	}
	catch (std::exception& e)
	{
		MessageBox(e.what());
		return;	
	}
}
void CGxSingleCamDlg::OnCbnSelchangeComboTriggerMode()
{
	// TODO: Add your control notification handler code here
	CComboBox* pCombo =NULL;
	try
	{
		int       nCurSel           = 0;
		gxstring  strCurrentText    = "";
		CString   strCurrentContent = "";       ///< The current value of combox

		pCombo = (CComboBox*)GetDlgItem(IDC_COMBO_TRIGGER_MODE);
		nCurSel = pCombo->GetCurSel();
		pCombo->GetLBText(nCurSel,strCurrentContent);
		strCurrentText = strCurrentContent.GetBuffer(0);

		//Set the value with combox to device
		m_objFeatureControlPtr->GetEnumFeature("TriggerMode")->SetValue(strCurrentText);
		m_nTriggerModeOld = nCurSel;
	}
	catch (CGalaxyException& e)
	{
		pCombo->SetCurSel(m_nTriggerModeOld);
		MessageBox(e.what());
		return;	
	}
	catch (std::exception& e)
	{
		pCombo->SetCurSel(m_nTriggerModeOld);
		MessageBox(e.what());
		return;	
	}
}

void CGxSingleCamDlg::OnCbnSelchangeComboTriggerSource()
{
	// TODO: Add your control notification handler code here
	CComboBox* pCombo =NULL;
	try
	{
		int       nCurSel           = 0;
		gxstring  strCurrentText    = "";
		CString   strCurrentContent = "";       ///< The current value of combox

		pCombo = (CComboBox*)GetDlgItem(IDC_COMBO_TRIGGER_SOURCE);
		nCurSel = pCombo->GetCurSel();
		pCombo->GetLBText(nCurSel,strCurrentContent);
		strCurrentText = strCurrentContent.GetBuffer(0);

		//Set the value with combox to device
		m_objFeatureControlPtr->GetEnumFeature("TriggerSource")->SetValue(strCurrentText);
		m_nTriggerSourceOld = nCurSel;
	}
	catch (CGalaxyException& e)
	{
		pCombo->SetCurSel(m_nTriggerSourceOld);
		MessageBox(e.what());
		return;	
	}
	catch (std::exception& e)
	{
		pCombo->SetCurSel(m_nTriggerSourceOld);
		MessageBox(e.what());
		return;	
	}
}

void CGxSingleCamDlg::OnCbnSelchangeComboTriggerActive()
{
	// TODO: Add your control notification handler code here
	CComboBox* pCombo =NULL;
	try
	{
		int       nCurSel           = 0;
		gxstring  strCurrentText    = "";
		CString   strCurrentContent = "";       ///< The current value of combox

		pCombo = (CComboBox*)GetDlgItem(IDC_COMBO_TRIGGER_ACTIVE);
		nCurSel = pCombo->GetCurSel();
		pCombo->GetLBText(nCurSel,strCurrentContent);
		strCurrentText = strCurrentContent.GetBuffer(0);

		//Set the value with combox to device
		m_objFeatureControlPtr->GetEnumFeature("TriggerActivation")->SetValue(strCurrentText);
		m_nTriggerActiveOld = nCurSel;
	}
	catch (CGalaxyException& e)
	{
		pCombo->SetCurSel(m_nTriggerActiveOld);
		MessageBox(e.what());
		return;	
	}
	catch (std::exception& e)
	{
		pCombo->SetCurSel(m_nTriggerActiveOld);
		MessageBox(e.what());
		return;	
	}
}

void CGxSingleCamDlg::OnCbnSelchangeComboBalanceWhiteAuto()
{
	// TODO: Add your control notification handler code here
	CComboBox* pCombo =NULL;
	try
	{
		int       nCurSel           = 0;
		gxstring  strCurrentText    = "";
		CString   strCurrentContent = "";      ///< The current value of combox

		pCombo = (CComboBox*)GetDlgItem(IDC_COMBO_BALANCE_WHITE_AUTO);
		nCurSel = pCombo->GetCurSel();
		pCombo->GetLBText(nCurSel,strCurrentContent);
		strCurrentText = strCurrentContent.GetBuffer(0);

		//Record the current value of auto white balance
		m_strBalanceWhiteAutoMode = strCurrentText;

		//Set the value with combox to device
		m_objFeatureControlPtr->GetEnumFeature("BalanceWhiteAuto")->SetValue(strCurrentText);

		if (strCurrentText == "Off")
		{
			GetDlgItem(IDC_EDIT_BALANCE_WHITE_SERISE)->EnableWindow(TRUE);
			m_dEditBalanceRatioValue = m_objFeatureControlPtr->GetFloatFeature("BalanceRatio")->GetValue();
			UpdateData(FALSE);
		}
		else
		{
			GetDlgItem(IDC_EDIT_BALANCE_WHITE_SERISE)->EnableWindow(FALSE);
		}

		m_nBalanceWhiteAutoOld = nCurSel;
	}
	catch (CGalaxyException& e)
	{
		pCombo->SetCurSel(m_nBalanceWhiteAutoOld);
		MessageBox(e.what());
		return;	
	}
	catch (std::exception& e)
	{
		pCombo->SetCurSel(m_nBalanceWhiteAutoOld);
		MessageBox(e.what());
		return;	
	}
}

void CGxSingleCamDlg::OnCbnSelchangeComboBalanceWhiteSelect()
{
	// TODO: Add your control notification handler code here
	CComboBox* pCombo =NULL;
	try
	{
		int       nCurSel           = 0;      
		gxstring  strCurrentText    = "";
		CString   strCurrentContent = "";        ///< The current value of combox

		pCombo = (CComboBox*)GetDlgItem(IDC_COMBO_BALANCE_WHITE_SELECT);
		nCurSel = pCombo->GetCurSel();
		pCombo->GetLBText(nCurSel,strCurrentContent);
		strCurrentText = strCurrentContent.GetBuffer(0);

		//Set the value with combox to device
		m_objFeatureControlPtr->GetEnumFeature("BalanceRatioSelector")->SetValue(strCurrentText);
		m_dEditBalanceRatioValue = m_objFeatureControlPtr->GetFloatFeature("BalanceRatio")->GetValue();
		UpdateData(FALSE);
		m_nBanlanceWhiteRatioOld = nCurSel;
	}
	catch (CGalaxyException& e)
	{
		pCombo->SetCurSel(m_nBanlanceWhiteRatioOld);
		MessageBox(e.what());
		return;	
	}
	catch (std::exception& e)
	{
		pCombo->SetCurSel(m_nBanlanceWhiteRatioOld);
		MessageBox(e.what());
		return;	
	}
}

void CGxSingleCamDlg::OnBnClickedCheckSaveBmp()
{
	// TODO: Add your control notification handler code here
		UpdateData(TRUE);
}

void CGxSingleCamDlg::OnClose()
{
	// TODO: Add your message handler code here and/or call default
	try
	{
		//Kill the timer
		KillTimer(0);

		//If the device is acquiring image then stop it.
		if (m_bIsSnap)
		{
			//Send AcquisitionStop command 
			m_objFeatureControlPtr->GetCommandFeature("AcquisitionStop")->Execute();

			//Stop stream capturing
			m_objStreamPtr->StopGrab();

			//Unregister the CaptureCallback function
			m_objStreamPtr->UnregisterCaptureCallback();

			m_bIsSnap = false;
		}
	}
	catch (CGalaxyException)
	{
		//do noting
	}
	catch (std::exception)
	{
		//do noting
	}

	try
	{
		//Check whether the device has been closed
		if (m_bIsOpen)
		{
			//Close stream
			m_objStreamPtr->Close();

			//Close device
			m_objDevicePtr->Close();

			m_bIsOpen = false;
		}
	}
	catch (CGalaxyException)
	{
		//do noting
	}
	catch (std::exception)
	{
		//do noting
	}

	try
	{
		//Release the resoources
		IGXFactory::GetInstance().Uninit();
	}
	catch (CGalaxyException)
	{
		//do noting
	}
	catch (std::exception)
	{
		//do noting
	}

	if (m_pSampleCaptureEventHandle != NULL)
	{
		delete m_pSampleCaptureEventHandle;
		m_pSampleCaptureEventHandle = NULL;
	}

	if (m_pBitmap != NULL)
	{
		delete m_pBitmap;
		m_pBitmap = NULL;
	}

	CDialog::OnClose();
}

void CGxSingleCamDlg::OnEnKillfocusEditShutter()
{
	// TODO: Add your control notification handler code here
	// Check whether the device has been closed
	if (!m_bIsOpen)
	{
		return;
	}
     
	double dShutterValueOld = m_dEditShutterValue;
	try
	{
		UpdateData(TRUE);

		//Check the input value whether is in the range of exposure time or not.
        //If the input value is bigger than the max value, it will be set the max value,otherwise is min value.
		if (m_dEditShutterValue > m_dShutterValueMax )
		{
			m_dEditShutterValue = m_dShutterValueMax;
		}
		if (m_dEditShutterValue < m_dShutterValueMin)
		{
			m_dEditShutterValue = m_dShutterValueMin;
		}

		m_objFeatureControlPtr->GetFloatFeature("ExposureTime")->SetValue(m_dEditShutterValue);
	}
	catch (CGalaxyException& e)
	{
		m_dEditShutterValue = dShutterValueOld;
		MessageBox(e.what());
	}
	catch (std::exception& e)
	{
		m_dEditShutterValue = dShutterValueOld;
		MessageBox(e.what());
	}

	UpdateData(FALSE);

}

void CGxSingleCamDlg::OnEnKillfocusEditGain()
{
	// TODO: Add your control notification handler code here
	//Check whether the device has been closed
	if (!m_bIsOpen)
	{
		return;
	}
    
	double dGainValueOld = m_dEditGainValue;
	try
	{
		UpdateData(TRUE);

		//Check the input value whether is in the range of gain or not.
        //If the input value is bigger than the max value, it will be set the max value,otherwise is min value.
		if (m_dEditGainValue > m_dGainValueMax )
		{
			m_dEditGainValue = m_dGainValueMax;
		}
		if (m_dEditGainValue < m_dGainValueMin)
		{
			m_dEditGainValue = m_dGainValueMin;
		}
		m_objFeatureControlPtr->GetFloatFeature("Gain")->SetValue(m_dEditGainValue);
	}
	catch (CGalaxyException& e)
	{
		m_dEditGainValue = dGainValueOld;
		MessageBox(e.what());
	}
	catch (std::exception& e)
	{
		m_dEditGainValue = dGainValueOld;
		MessageBox(e.what());
	}

	UpdateData(FALSE);
}

void CGxSingleCamDlg::OnEnKillfocusEditBalanceWhiteSerise()
{
	// TODO: Add your control notification handler code here
	//Check whether the device has been closed
	if (!m_bIsOpen)
	{
		return;
	}

    double dBalanceWhiteRatioOld = m_dEditBalanceRatioValue;
	try
	{
		UpdateData(TRUE);

		//Check the input value whether is in the range of white balance ratio or not.
        //If the input value is bigger than the max value, it will be set the max value,otherwise is min value.
		if (m_dEditBalanceRatioValue > m_dBalanceWhiteRatioMax)
		{
			m_dEditBalanceRatioValue = m_dBalanceWhiteRatioMax;
		}
		if ((m_dEditBalanceRatioValue < m_dBalanceWhiteRatioMin))
		{
			m_dEditBalanceRatioValue = m_dBalanceWhiteRatioMin;
		}
		m_objFeatureControlPtr->GetFloatFeature("BalanceRatio")->SetValue(m_dEditBalanceRatioValue);
	}
	catch (CGalaxyException& e)
	{
		m_dEditBalanceRatioValue = dBalanceWhiteRatioOld;
		MessageBox(e.what());
	}
	catch (std::exception& e)
	{
		m_dEditBalanceRatioValue = dBalanceWhiteRatioOld;
		MessageBox(e.what());
	}

	UpdateData(FALSE);
}

void CGxSingleCamDlg::OnTimer(UINT_PTR nIDEvent)
{
	// TODO: Add your message handler code here and/or call default
	try
	{
		CString strText  = "";    // the content widget will display
		CComboBox* pComBox = (CComboBox*)GetDlgItem(IDC_COMBO_BALANCE_WHITE_AUTO);


		//When setting the value of once to device,after setting successfuly, it will became off  automatically by the device itself.
		//The program update the content of UI through reading the white balance current value from device regularly
		if (m_strBalanceWhiteAutoMode == "Once")
		{
			//Get the value of auto white balance
			m_strBalanceWhiteAutoMode = m_objFeatureControlPtr->GetEnumFeature("BalanceWhiteAuto")->GetValue();
			GxIAPICPP::gxstring_vector vecBalanceWhiteAutoEnumCount;
			vecBalanceWhiteAutoEnumCount = m_objFeatureControlPtr->GetEnumFeature("BalanceWhiteAuto")
				                                                 ->GetEnumEntryList();

			//Check whether the auto white balance value is off or not
			if (m_strBalanceWhiteAutoMode == "Off")
			{
				for (uint32_t i = 0; i< vecBalanceWhiteAutoEnumCount.size(); i++)
				{
					pComBox->GetLBText(i,strText);
					if (strText == "Off")
					{
						// Select the item of off in the auto white balance UI widget, that is from once to off.
						pComBox->SetCurSel(i);
						GetDlgItem(IDC_EDIT_BALANCE_WHITE_SERISE)->EnableWindow(TRUE);
						m_dEditBalanceRatioValue = m_objFeatureControlPtr->GetFloatFeature("BalanceRatio")->GetValue();
						UpdateData(FALSE);
						break;
					}
				}
			}
		}
	}
	catch (CGalaxyException)
	{
		return;
	}
	catch (std::exception)
	{
		return;
	}
	CDialog::OnTimer(nIDEvent);
}

//----------------------------------------------------------------------------------
/**
\brief   Save the image as BMP format
\param   image information

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamDlg::SavePicture(CImageDataPointer& objImageDataPointer)
{
	try
	{
		SYSTEMTIME   sysTime;                   ///< system time
		CString      strFileName = "";          ///< the name of path to saving image
		CString      strFilePath = "";          ///< the file path
		strFilePath = m_strSavePath.c_str();

		//Create the folder to saving images.
		BOOL bRet = CreateDirectory(strFilePath,NULL);

		//Get the current system time as the default image name will be saved.
		GetLocalTime(&sysTime);
		strFileName.Format("%s\\%d_%d_%d_%d_%d_%d_%d.bmp",strFilePath,  
			                                              sysTime.wYear,
			                                              sysTime.wMonth,
			                                              sysTime.wDay,
			                                              sysTime.wHour,
			                                              sysTime.wMinute,
			                                              sysTime.wSecond,
			                                              sysTime.wMilliseconds);

		//Save image as BMP
		m_pBitmap->SaveBmp(objImageDataPointer,strFileName.GetBuffer(0));
	}
	catch (std::exception)
	{
		//Due to saving image is completed in the capture callback function of the acquisition thread, if it throw
		//exception,the acquisition thread will be stoped. So in order to avoid stoping the thread, in the process of saving
		//image doesn't throw exception.
		return;
		
	}

}
BOOL CGxSingleCamDlg::PreTranslateMessage(MSG* pMsg)
{
	CWnd  *pWnd   = NULL;
	int   nCtrlID = 0;             //< save the widget ID 

	//Check whether is the message sent by enter key. 
	if ((pMsg->message == WM_KEYDOWN) && (pMsg->wParam == VK_RETURN))   
	{    
		//Get the window (or widget) pointer whitch has the input focus.
		pWnd = GetFocus();

		//Get the widget ID that has the current focus.
		nCtrlID = pWnd->GetDlgCtrlID();

		//Check the type of ID.
		switch(nCtrlID)
		{
		case IDC_EDIT_SHUTTER:
		case IDC_EDIT_GAIN:
		case IDC_EDIT_BALANCE_WHITE_SERISE:

			//lose the focus.
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
