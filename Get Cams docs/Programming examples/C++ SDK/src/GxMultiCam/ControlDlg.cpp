// ControlDlg.cpp : implementation file
//

#include "stdafx.h"
#include "GxMultiCam.h"
#include "ControlDlg.h"
#include "GxMultiCamDoc.h"

using namespace std;

// CControlDlg

IMPLEMENT_DYNCREATE(CControlDlg, CFormView)

CControlDlg::CControlDlg()
	: CFormView(CControlDlg::IDD)
	, m_dEditShutter(0)
	, m_dEditGain(0)
	, m_nEditSnapSpeed(0)
	, m_bCheckShowDevice(FALSE)
{
	m_bIsSnapSpeed                = false;
	m_bIsColorFilter              = false;
	m_dShutterMax                 = 0;
	m_dShutterMin                 = 0;
	m_dGainMax                    = 0;
	m_dGainMin                    = 0;
	m_nSnapSpeedMax               = 0;
	m_nSnapSpeedMin               = 0;
	m_nDeviceListCurrent          = 0;
	m_nCurrentBalanceAutoWhiteSel = 0;
	m_strBalanceWhiteAuto         = "Off";
	m_pDeviceProcessCurrent       = NULL;

	for (int i = 0; i< DEVICE_CONTS; i++)
	{
		m_pDeviceProcess[i]          = NULL;
	}

}

CControlDlg::~CControlDlg()
{
}

void CControlDlg::DoDataExchange(CDataExchange* pDX)
{
	CFormView::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_COMBO_DEVICE_LIST, m_comboxDevice);
	DDX_Control(pDX, IDC_COMBO_BALANCE_WHITE_AUTO, m_comboxBalanceWhiteAuto);
	DDX_Control(pDX, IDC_STATIC_SHUTTER_TIME, m_staticShutterTime);
	DDX_Control(pDX, IDC_STATIC_GAIN, m_staticGain);
	DDX_Control(pDX, IDC_STATIC_SNAP_SPEED, m_staticSnapSpeed);
	DDX_Control(pDX, IDC_EDIT_SHUTTER, m_editShutter);
	DDX_Control(pDX, IDC_EDIT_GAIN, m_editGain);
	DDX_Text(pDX, IDC_EDIT_SHUTTER, m_dEditShutter);
	DDX_Text(pDX, IDC_EDIT_GAIN, m_dEditGain);
	DDX_Text(pDX, IDC_EDIT_SNAP_SPEED, m_nEditSnapSpeed);
	DDX_Check(pDX, IDC_CHECK_SHOW_DEVICE_FPS, m_bCheckShowDevice);
}

BEGIN_MESSAGE_MAP(CControlDlg, CFormView)
	ON_BN_CLICKED(IDC_BTN_OPEN_DEVICE, &CControlDlg::OnBnClickedBtnOpenDevice)
	ON_BN_CLICKED(IDC_BTN_START_SNAP, &CControlDlg::OnBnClickedBtnStartSnap)
	ON_BN_CLICKED(IDC_BTN_STOP_SNAP, &CControlDlg::OnBnClickedBtnStopSnap)
	ON_BN_CLICKED(IDC_BTN_CLOSE_DEVICE, &CControlDlg::OnBnClickedBtnCloseDevice)
	ON_CBN_SELCHANGE(IDC_COMBO_DEVICE_LIST, &CControlDlg::OnCbnSelchangeComboDeviceList)
	ON_CBN_SELCHANGE(IDC_COMBO_BALANCE_WHITE_AUTO, &CControlDlg::OnCbnSelchangeComboBalanceWhiteAuto)
	ON_EN_KILLFOCUS(IDC_EDIT_SHUTTER, &CControlDlg::OnEnKillfocusEditShutter)
	ON_EN_KILLFOCUS(IDC_EDIT_GAIN, &CControlDlg::OnEnKillfocusEditGain)
	ON_EN_KILLFOCUS(IDC_EDIT_SNAP_SPEED, &CControlDlg::OnEnKillfocusEditSnapSpeed)
	ON_WM_TIMER()
	ON_CBN_DROPDOWN(IDC_COMBO_DEVICE_LIST, &CControlDlg::OnCbnDropdownComboDeviceList)
	ON_BN_CLICKED(IDC_CHECK_SHOW_DEVICE_FPS, &CControlDlg::OnBnClickedCheckShowDeviceFps)
	ON_WM_DESTROY()
END_MESSAGE_MAP()


// CControlDlg diagnostics

#ifdef _DEBUG
void CControlDlg::AssertValid() const
{
	CFormView::AssertValid();
}

#ifndef _WIN32_WCE
void CControlDlg::Dump(CDumpContext& dc) const
{
	CFormView::Dump(dc);
}
#endif
#endif //_DEBUG

void CControlDlg::OnInitialUpdate()
{
	CFormView::OnInitialUpdate();
	try
	{
		for (int i=0; i<DEVICE_CONTS; i++)
		{
			m_pDeviceProcess[i] = new CDeviceProcess;
		}
		// Initialize GxIAPICPP
		IGXFactory::GetInstance().Init();

		// Enumerate all camera devices
		IGXFactory::GetInstance().UpdateDeviceList(1000,m_vectorDeviceInfo);

		// Check whether devices has been detected
		if (m_vectorDeviceInfo.size() <= 0)
		{
			MessageBox("No device found,please make sure the device is properly connected and then restart the program!");
            GetDlgItem(IDC_COMBO_DEVICE_LIST)->EnableWindow(false);
			GetDlgItem(IDC_BTN_OPEN_DEVICE)->EnableWindow(false);
			GetDlgItem(IDC_BTN_CLOSE_DEVICE)->EnableWindow(false);
			GetDlgItem(IDC_BTN_START_SNAP)->EnableWindow(false);
			GetDlgItem(IDC_BTN_STOP_SNAP)->EnableWindow(false);
			GetDlgItem(IDC_EDIT_SHUTTER)->EnableWindow(false);
			GetDlgItem(IDC_EDIT_GAIN)->EnableWindow(false);
			GetDlgItem(IDC_EDIT_SNAP_SPEED)->EnableWindow(false);
			GetDlgItem(IDC_COMBO_BALANCE_WHITE_AUTO)->EnableWindow(false);
			GetDlgItem(IDC_CHECK_SHOW_DEVICE_FPS)->EnableWindow(false);
			CMainFrame* pMainFrm = (CMainFrame *)AfxGetApp()->GetMainWnd();
			pMainFrm->m_objControlDlg = this;
			return;
		}

		// Display device name to the device list
		for (uint32_t i =0; i< m_vectorDeviceInfo.size(); i++)
		{
			// Check whether the number of device is greater than four
			// If the number of devices is greater than four, only four pieces of device information are reserved
			if (i >= DEVICE_CONTS)
			{
				break;
			}

			gxstring strDeviceInformation = "";
			strDeviceInformation =m_vectorDeviceInfo[i].GetDisplayName();
			m_comboxDevice.SetItemData(m_comboxDevice.AddString(strDeviceInformation.c_str()),i);
			m_mapDeviceInformation.insert(map<int,CDeviceProcess*>::value_type(i,m_pDeviceProcess[i]));

			// Update UI
			__UpdateUI(m_pDeviceProcess[i]);

		}

		// Set Timer
		SetTimer(0,1000,NULL);
		m_comboxDevice.SetCurSel(0);

		// Get the currently displayed device handling class object
		m_pDeviceProcessCurrent = m_mapDeviceInformation[0];

		CMainFrame * pMainFrm = (CMainFrame *)AfxGetApp()->GetMainWnd();
		pMainFrm->m_objControlDlg = this;
		
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
	
	// TODO: Add your specialized code here and/or call the base class
	
}

// CControlDlg message handlers

void CControlDlg::OnBnClickedBtnOpenDevice()
{
	// TODO: Add your control notification handler code here

    try
    {
		// Get the child window handle of displaying area
		CWnd*  pWnd = NULL;
		CMainFrame * pMainFrm = (CMainFrame *)AfxGetApp()->GetMainWnd();
		switch(m_nDeviceListCurrent)
		{
		case 0:
			pWnd = pMainFrm->m_wndSplitterLeft.GetPane(0, 0);
			break;
		case 1:
			pWnd = pMainFrm->m_wndSplitterLeft.GetPane(0, 1);
			break;
		case 2:
			pWnd = pMainFrm->m_wndSplitterLeft.GetPane(1, 0);
			break;
		case 3:
			pWnd = pMainFrm->m_wndSplitterLeft.GetPane(1, 1);
			break;

		default: break;
		}

		// Get device index
		int nDeviceIndex = m_nDeviceListCurrent+1;

		// Open device
		m_pDeviceProcessCurrent->OpenDevice(m_vectorDeviceInfo[m_nDeviceListCurrent].GetSN(),pWnd,nDeviceIndex);

		// initialize UI
		__InitUI(m_pDeviceProcessCurrent->m_objFeatureControlPtr);

		// Update UI
		__UpdateUI(m_pDeviceProcessCurrent);

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
//----------------------------------------------------------------------
/**
\brief  initialize UI

\param  objFeatureControlPtr[in]      Feature control

\return void
**/
//----------------------------------------------------------------------
void CControlDlg::__InitUI(CGXFeatureControlPointer objFeatureControlPtr)
{
 
	if (objFeatureControlPtr.IsNull())
	{
		return;
	}
    
	bool bIsBalanceWhiteAutoRead = false;        // The flag indicate whether white balance can be read 
	bool bBalanceWhiteAuto       = false;        // The flag indicate whether white balance is supported

	// The flag indicate whether Bayer format is supported
	m_bIsColorFilter = objFeatureControlPtr->IsImplemented("PixelColorFilter");

	if (m_bIsColorFilter)
	{
		// whether white balance is supported
		bBalanceWhiteAuto = objFeatureControlPtr->IsImplemented("BalanceWhiteAuto");

		// whether white balance can be read 
		bIsBalanceWhiteAutoRead = objFeatureControlPtr->IsReadable("BalanceWhiteAuto");
		if (bBalanceWhiteAuto)
		{
			if (bIsBalanceWhiteAutoRead)
			{
				m_strBalanceWhiteAuto = objFeatureControlPtr->GetEnumFeature("BalanceWhiteAuto")->GetValue();
			}
			int nCursel  = 0;
			gxstring strCurEnumList = "";
			GxIAPICPP::gxstring_vector vectorEnumEntryList;

			// Get the current device function value
			strCurEnumList = objFeatureControlPtr->GetEnumFeature("BalanceWhiteAuto")->GetValue();

			// Get all enumerations for this feature
			vectorEnumEntryList = objFeatureControlPtr->GetEnumFeature("BalanceWhiteAuto")->GetEnumEntryList();

			// Clear device list
			m_comboxBalanceWhiteAuto.ResetContent();
			for (uint32_t i = 0; i<vectorEnumEntryList.size(); i++)
			{
				std::string strEnumList = vectorEnumEntryList[i].c_str();
				m_comboxBalanceWhiteAuto.SetItemData(m_comboxBalanceWhiteAuto.AddString(strEnumList.c_str()),i);

				if (strCurEnumList == vectorEnumEntryList[i])
				{
					nCursel = i;
				}

			}
			m_nCurrentBalanceAutoWhiteSel = nCursel;
			m_comboxBalanceWhiteAuto.SetCurSel(nCursel);
		}
	}
	

	// Initialize exposure time
	CString  strShutterTimeRange = "";          // Exposure time range
	gxstring strShutterTimeUint  = "";          // Exposure time unit

	m_dEditShutter     = objFeatureControlPtr->GetFloatFeature("ExposureTime")->GetValue();
	strShutterTimeUint = objFeatureControlPtr->GetFloatFeature("ExposureTime")->GetUnit();
	m_dShutterMax      = objFeatureControlPtr->GetFloatFeature("ExposureTime")->GetMax();
	m_dShutterMin      = objFeatureControlPtr->GetFloatFeature("ExposureTime")->GetMin();

	strShutterTimeRange.Format("Exposure time(%.4f~%.4f)%s",m_dShutterMin,m_dShutterMax,strShutterTimeUint.c_str());
	m_staticShutterTime.SetWindowText(strShutterTimeRange);

	// Initialize gain
	CString  strGainRange = "";                // Gain range
	gxstring strGainUint  = "";                // Gain unit

	m_dEditGain   = objFeatureControlPtr->GetFloatFeature("Gain")->GetValue();
	strGainUint   = objFeatureControlPtr->GetFloatFeature("Gain")->GetUnit();
	m_dGainMax    = objFeatureControlPtr->GetFloatFeature("Gain")->GetMax();
	m_dGainMin    = objFeatureControlPtr->GetFloatFeature("Gain")->GetMin();

	strGainRange.Format("Gain(%.4f~%.4f)%s",m_dGainMin,m_dGainMax,strGainUint.c_str());
	m_staticGain.SetWindowText(strGainRange);

	// Initialize acquisition speed level
	// whether acquisition speed level can be read 
	m_bIsSnapSpeed = objFeatureControlPtr->IsImplemented("AcquisitionSpeedLevel");
	if (m_bIsSnapSpeed)
	{
		CString  strSnapSpeedRange = "";  
		m_nEditSnapSpeed   = objFeatureControlPtr->GetIntFeature("AcquisitionSpeedLevel")->GetValue();
		m_nSnapSpeedMax    = objFeatureControlPtr->GetIntFeature("AcquisitionSpeedLevel")->GetMax();
		m_nSnapSpeedMin    = objFeatureControlPtr->GetIntFeature("AcquisitionSpeedLevel")->GetMin();

		strSnapSpeedRange.Format("AcquisitionSpeedLevel(%lld~%lld)",m_nSnapSpeedMin,m_nSnapSpeedMax);
		m_staticSnapSpeed.SetWindowText(strSnapSpeedRange);

	}
	else
	{
		// Update Edit
		m_nEditSnapSpeed = NULL;
	}

	// Get whether the frame rate flag is displayed
	m_bCheckShowDevice = m_pDeviceProcessCurrent->GetShowFPS();

	UpdateData(FALSE);

}

//----------------------------------------------------------------------
/**
\brief  Update UI

\param  pDeviceProcess[in]     the object of device process class 

\return void
**/
//----------------------------------------------------------------------
void CControlDlg::__UpdateUI(CDeviceProcess* pDeviceProcess)
{
	if (pDeviceProcess == NULL)
	{
		return;
	}

	bool IsOpen = pDeviceProcess->IsOpen();
	bool IsSnap = pDeviceProcess->IsSnap();
	GetDlgItem(IDC_BTN_OPEN_DEVICE)->EnableWindow(!IsOpen);
	GetDlgItem(IDC_BTN_CLOSE_DEVICE)->EnableWindow(IsOpen);
	GetDlgItem(IDC_BTN_START_SNAP)->EnableWindow(IsOpen&& !IsSnap);
	GetDlgItem(IDC_BTN_STOP_SNAP)->EnableWindow(IsOpen&& IsSnap);
	GetDlgItem(IDC_EDIT_SHUTTER)->EnableWindow(IsOpen);
	GetDlgItem(IDC_EDIT_GAIN)->EnableWindow(IsOpen);
	GetDlgItem(IDC_EDIT_SNAP_SPEED)->EnableWindow(IsOpen&& !IsSnap&& m_bIsSnapSpeed);
	GetDlgItem(IDC_COMBO_BALANCE_WHITE_AUTO)->EnableWindow(IsOpen&& m_bIsColorFilter);
	GetDlgItem(IDC_CHECK_SHOW_DEVICE_FPS)->EnableWindow(IsOpen);

}

void CControlDlg::OnBnClickedBtnStartSnap()
{
	// TODO: Add your control notification handler code here
	try
	{
		// Start acquisition
		m_pDeviceProcessCurrent->StartSnap();

		// Update UI
		__UpdateUI(m_pDeviceProcessCurrent);

		// Update equipment model and frame rate information
		m_pDeviceProcessCurrent->RefreshDeviceSNFPS();

		SetTimer(1, 1000, NULL);
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

void CControlDlg::OnBnClickedBtnStopSnap()
{
	// TODO: Add your control notification handler code here
	try
	{
		// Stop acquisition
		m_pDeviceProcessCurrent->StopSnap();

		// Update UI
		__UpdateUI(m_pDeviceProcessCurrent);

		//Turn off timer 1
		KillTimer(1);
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

void CControlDlg::OnBnClickedBtnCloseDevice()
{
	// TODO: Add your control notification handler code here
	try
	{
		// Close device
		m_pDeviceProcessCurrent->CloseDevice();
		

		// Update UI
		__UpdateUI(m_pDeviceProcessCurrent);
	}
	catch (CGalaxyException)
	{
		//do noting
	}
	catch (std::exception)
	{
		//do noting
		return;
	}
}

void CControlDlg::OnCbnSelchangeComboDeviceList()
{
	// TODO: Add your control notification handler code here
	try
	{
		m_nDeviceListCurrent = m_comboxDevice.GetCurSel();

		m_pDeviceProcessCurrent = m_mapDeviceInformation[m_nDeviceListCurrent];

		// Check whether the device has been opened
		bool bIsOpen = m_pDeviceProcessCurrent->IsOpen();
		if (!bIsOpen)
		{
			__UpdateUI(m_pDeviceProcessCurrent);
		}

		else
		{
			// Initialize UI
			__InitUI(m_pDeviceProcessCurrent->m_objFeatureControlPtr);

			// Update UI
			__UpdateUI(m_pDeviceProcessCurrent);

		}
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

void CControlDlg::OnCbnSelchangeComboBalanceWhiteAuto()
{
	// TODO: Add your control notification handler code here
	try
	{
		
		int      nCurSelBalanceWhiteAuto = 0;      // The current value of the  auto white balance ComboBox
		CString  strCurText              = "";     // The current value of the  device auto white balance
		gxstring strCurEnumList          = "";     

		nCurSelBalanceWhiteAuto = m_comboxBalanceWhiteAuto.GetCurSel();
		m_comboxBalanceWhiteAuto.GetLBText(nCurSelBalanceWhiteAuto,strCurText);
		strCurEnumList = strCurText.GetBuffer(0);

		// Record current white balance mode
		m_strBalanceWhiteAuto = strCurEnumList;

		m_pDeviceProcessCurrent->m_objFeatureControlPtr->GetEnumFeature("BalanceWhiteAuto")->SetValue(strCurEnumList); 

		// Record current value of the ComboBox
		m_nCurrentBalanceAutoWhiteSel =nCurSelBalanceWhiteAuto;

	}
	catch (CGalaxyException& e)
	{
		m_comboxBalanceWhiteAuto.SetCurSel(m_nCurrentBalanceAutoWhiteSel);
		MessageBox(e.what());
		return;
	}
	catch (std::exception& e)
	{
		m_comboxBalanceWhiteAuto.SetCurSel(m_nCurrentBalanceAutoWhiteSel);
		MessageBox(e.what());
		return;
	}
}

void CControlDlg::OnEnKillfocusEditShutter()
{
	// TODO: Add your control notification handler code here
	double dShutterOld = m_dEditShutter;           // Record exposure time
 	if (!m_pDeviceProcessCurrent->IsOpen())
	{
		return;
	}
	try
	{
		UpdateData(TRUE);

		// Check whether the input value is within the exposure time range
		if (m_dEditShutter < m_dShutterMin  )
		{
			m_dEditShutter = m_dShutterMin;
		}

		if (m_dEditShutter > m_dShutterMax)
		{
			m_dEditShutter = m_dShutterMax;
		}

		// Set this value to device
		m_pDeviceProcessCurrent->m_objFeatureControlPtr->GetFloatFeature("ExposureTime")->SetValue(m_dEditShutter); 

	}
	catch (CGalaxyException& e)
	{
		m_dEditShutter = dShutterOld;
		MessageBox(e.what());
	}
	catch (std::exception& e)
	{
		m_dEditShutter = dShutterOld;
		MessageBox(e.what());
	}

	UpdateData(FALSE);
}

void CControlDlg::OnEnKillfocusEditGain()
{
	// TODO: Add your control notification handler code here
	double dGainOld = m_dEditGain;         // Record Gain value
	if (!m_pDeviceProcessCurrent->IsOpen())
	{
		return;
	}

	try
	{
		UpdateData(TRUE);

		// Check whether the input value is within the gain range
		if (m_dEditGain < m_dGainMin )
		{
			m_dEditGain = m_dGainMin;
		}

		if (m_dEditGain > m_dGainMax)
		{
			m_dEditGain = m_dGainMax;
		}

		// Set this value to device
		m_pDeviceProcessCurrent->m_objFeatureControlPtr->GetFloatFeature("Gain")->SetValue(m_dEditGain); 

	}
	catch (CGalaxyException& e)
	{
		m_dEditGain = dGainOld;
		MessageBox(e.what());
	}
	catch (std::exception& e)
	{
		m_dEditGain = dGainOld;
		MessageBox(e.what());
	}

	UpdateData(FALSE);
}

void CControlDlg::OnEnKillfocusEditSnapSpeed()
{
	// TODO: Add your control notification handler code here
	int64_t     nSnapSpeedOld    = m_nEditSnapSpeed;      // Record acquisition speed level

	try
	{
		UpdateData(TRUE);

		// Check whether the input value is within the acquisition speed level range
		if (m_nEditSnapSpeed < m_nSnapSpeedMin  )
		{
			m_nEditSnapSpeed = m_nSnapSpeedMin;
		}

		if (m_nEditSnapSpeed > m_nSnapSpeedMax)
		{
			m_nEditSnapSpeed = m_nSnapSpeedMax;
		}

		// Set this value to device
		m_pDeviceProcessCurrent->m_objFeatureControlPtr->GetIntFeature("AcquisitionSpeedLevel")->SetValue(m_nEditSnapSpeed); 

	}
	catch (CGalaxyException& e)
	{
		m_nEditSnapSpeed = nSnapSpeedOld;
		MessageBox(e.what());
	}
	catch (std::exception& e)
	{
		m_nEditSnapSpeed = nSnapSpeedOld;
		MessageBox(e.what());
	}

	UpdateData(FALSE);
}

void CControlDlg::OnBnClickedCheckShowDeviceFps()
{
	// TODO: Add your control notification handler code here
	try
	{
		UpdateData(TRUE);
		m_pDeviceProcessCurrent->SetShowFPS(m_bCheckShowDevice == TRUE);

		if (1 == ((CButton *)GetDlgItem(IDC_CHECK_SHOW_DEVICE_FPS))->GetCheck())
		{
			SetTimer(1, 1000, NULL);
		}
		else
		{
			KillTimer(1);
		}
	}
	catch (std::exception& e)
	{
		MessageBox(e.what());
		return;
	}
	
}

BOOL CControlDlg::PreTranslateMessage(MSG* pMsg)
{
	CWnd  *pWnd   = NULL;
	int   nCtrlID = 0;             

	// Check whether the keyboard enter the message
	if ((pMsg->message == WM_KEYDOWN) && (pMsg->wParam == VK_RETURN))   
	{    
		pWnd = GetFocus();

		// Get current control focus
		nCtrlID = pWnd->GetDlgCtrlID();

		switch(nCtrlID)
		{
		case IDC_EDIT_SHUTTER:
		case IDC_EDIT_GAIN:
		case IDC_EDIT_SNAP_SPEED:

			// Lose focus
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
	return CFormView::PreTranslateMessage(pMsg);
}

void CControlDlg::OnTimer(UINT_PTR nIDEvent)
{
	// TODO: Add your message handler code here and/or call default

	switch(nIDEvent)
	{
	case 0:
		try
		{
			// If the automatic white balance mode is Once, the actual white balance mode will automatically turn off after the setting is successful
			// The program updates the UI interface by reading the current value of the internal white balance of the device on a regular basis
			CString strCurText = "";
			if (m_strBalanceWhiteAuto == "Once")
			{
				// Get the current white balance of the device
				m_strBalanceWhiteAuto = m_pDeviceProcessCurrent->m_objFeatureControlPtr->GetEnumFeature("BalanceWhiteAuto")
					->GetValue();
				// Get the device white balance enumeration
				GxIAPICPP::gxstring_vector vectorEnumEntryList;
				vectorEnumEntryList = m_pDeviceProcessCurrent->m_objFeatureControlPtr->GetEnumFeature("BalanceWhiteAuto")
					->GetEnumEntryList();
				// Determine whether the white balance enumeration value is Off
				if (m_strBalanceWhiteAuto == "Off")
				{
					for (uint32_t i = 0; i < vectorEnumEntryList.size(); i++)
					{
						m_comboxBalanceWhiteAuto.GetLBText(i,strCurText);
						if (strCurText == "Off")
						{
							m_comboxBalanceWhiteAuto.SetCurSel(i);
							break;
						}
					}

				}

			}
			CFormView::OnTimer(nIDEvent);
		}
		catch (CGalaxyException)
		{
			return;
		}
		catch (std::exception)
		{
			return;
		}

		break;
	case 1:
		// Traverse the opened device and set the refresh FPS operation
		for (map<int,CDeviceProcess*>::iterator it = m_mapDeviceInformation.begin(); 
			it != m_mapDeviceInformation.end(); ++it)
		{
			it->second->RefreshDeviceSNFPS();
		}
		break;

	default:
		break;
	}
	CFormView::OnTimer(nIDEvent);
}

void CControlDlg::OnCbnDropdownComboDeviceList()
{
	// TODO: Add your control notification handler code here
	CComboBox *pWnd = (CComboBox*)GetDlgItem(IDC_COMBO_DEVICE_LIST);
	int nNumEntries = pWnd->GetCount();
	int nWidth = 0;
	CString str;

	CClientDC dc(pWnd);
	int nSave = dc.SaveDC();
	dc.SelectObject(GetFont());

	int nScrollWidth = ::GetSystemMetrics(SM_CXVSCROLL);
	for (int i = 0; i < nNumEntries; i++)
	{
		pWnd->GetLBText(i, str);
		int nLength = dc.GetTextExtent(str).cx + nScrollWidth;
		nWidth = max(nWidth, nLength);
	}

	// Add margin space to the calculations
	nWidth += dc.GetTextExtent("0").cx;

	dc.RestoreDC(nSave);
	pWnd->SetDroppedWidth(nWidth);
}

void CControlDlg::OnDestroy()
{
	try
	{
		// Close all device
		map<int,CDeviceProcess*>::iterator it;
		for (it = m_mapDeviceInformation.begin(); it != m_mapDeviceInformation.end(); ++it)
		{
			it->second->CloseDevice();

		}
		m_mapDeviceInformation.clear();

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
		// Release GxIAPICPP
		IGXFactory::GetInstance().Uninit();
	}
	catch (CGalaxyException)
	{
		//do noting
	}



	for (int i = 0; i <DEVICE_CONTS; i++)
	{
		if (m_pDeviceProcess[i] != NULL)
		{
			delete m_pDeviceProcess[i];
			m_pDeviceProcess[i] = NULL;
		}
	}
	CFormView::OnDestroy();

	// TODO: Add your message handler code here
}
