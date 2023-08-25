#pragma once
#include <iostream>
#include "DeviceProcess.h"
#include "MainFrm.h"
#include "afxwin.h"
#include "map"

#define  DEVICE_CONTS 4               ///< Up to four devices can be operated at the same time

// CControlDlg form view

class CControlDlg : public CFormView
{
	DECLARE_DYNCREATE(CControlDlg)

public:
	CControlDlg();           // protected constructor used by dynamic creation
	virtual ~CControlDlg();

public:
	enum { IDD = IDD_CONTROL_DIALOG };
#ifdef _DEBUG
	virtual void AssertValid() const;
#ifndef _WIN32_WCE
	virtual void Dump(CDumpContext& dc) const;
#endif
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedBtnOpenDevice();
	afx_msg void OnBnClickedBtnStartSnap();
	afx_msg void OnBnClickedBtnStopSnap();
	afx_msg void OnBnClickedBtnCloseDevice();
	afx_msg void OnCbnSelchangeComboDeviceList();
	afx_msg void OnCbnSelchangeComboBalanceWhiteAuto();
	afx_msg void OnEnKillfocusEditShutter();
	afx_msg void OnEnKillfocusEditGain();
	afx_msg void OnEnKillfocusEditSnapSpeed();

public:
	virtual BOOL PreTranslateMessage(MSG* pMsg);
	virtual void OnInitialUpdate();
	afx_msg void OnTimer(UINT_PTR nIDEvent);
	afx_msg void OnCbnDropdownComboDeviceList();
	afx_msg void OnBnClickedCheckShowDeviceFps();
	BOOL      m_bCheckShowDevice;
	CComboBox m_comboxDevice;
	CComboBox m_comboxBalanceWhiteAuto;
	CStatic   m_staticShutterTime;
	CStatic   m_staticGain;
	CStatic   m_staticSnapSpeed;
	CEdit     m_editShutter;
	CEdit     m_editGain;
	CEdit     m_editSnapSpeed;
	double    m_dEditShutter;
	double    m_dEditGain;
	int64_t   m_nEditSnapSpeed;

private:
	
	/// Initialize UI
	void __InitUI(CGXFeatureControlPointer objFeatureControlPtr);

	/// Update UI
	void __UpdateUI(CDeviceProcess* pDeviceProcess);

	GxIAPICPP::gxdeviceinfo_vector m_vectorDeviceInfo;          ///< Device information
	CDeviceProcess* m_pDeviceProcess[DEVICE_CONTS];             ///< The pointer for device manager
	CDeviceProcess* m_pDeviceProcessCurrent;                    ///< The pointer for current device process
	std::map<int, CDeviceProcess*> m_mapDeviceInformation;      ///< Storage device processing class
	bool     m_bIsSnapSpeed;                                    ///< The flag indicates whether the device supports SnapSpeed
	bool     m_bIsColorFilter;                                  ///< The flag indicates whether the device supports Bayer format
	double   m_dShutterMax;                                     ///< Maximum Shutter
	double   m_dShutterMin;                                     ///< Minimum Shutter
	double   m_dGainMax;                                        ///< Maximum Gain
	double   m_dGainMin;                                        ///< Minimum Gain
	int64_t  m_nSnapSpeedMax;                                   ///< Maximum SnapSpeed
	int64_t  m_nSnapSpeedMin;                                   ///< Minimum SnapSpeed
	int      m_nDeviceListCurrent;                              ///< Current device list index
	int      m_nCurrentBalanceAutoWhiteSel;                     ///< Record Current BalanceAutoWhite value
	gxstring m_strBalanceWhiteAuto;                             ///< Record BalanceWhiteAuto value

public:
	afx_msg void OnDestroy();
};


