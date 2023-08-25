// GxFlatFieldCorrectionDlg.h : header file
//

#if !defined(AFX_GxFlatFieldCorrectionDLG_H__D8D963C2_3BE2_44EC_B0EB_FCE016DDD6FA__INCLUDED_)
#define AFX_GxFlatFieldCorrectionDLG_H__D8D963C2_3BE2_44EC_B0EB_FCE016DDD6FA__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#define  WHITE_BALANCE_RNTRY               3      ///< The enumeration entry of BalanceWhiteAuto

#define  GRAY_MIN                          0      ///< Minimum gray value
#define  GRAY_MAX                          255    ///< Maximum gray value
#define  GRAY_INVALID                      256    ///< Invalid gray value
#define  GRAY_INIT                         127    ///< Initial gray value
#define  BRIGHT_IMG_GRAY_MIN               20     ///< Minimum gray value of bright field image
#define  BRIGHT_IMG_GRAY_MAX               250    ///< Maximum gray value of bright field image

#define  CORRECTION_UI_FRAME_COUNT_MIN     0      ///< Minimum acquisition frame count
#define  CORRECTION_UI_FRAME_COUNT_1       1      ///< Acquisition frame count 1 of the UI setting
#define  CORRECTION_UI_FRAME_COUNT_2       2      ///< Acquisition frame count 2 of the UI setting
#define  CORRECTION_UI_FRAME_COUNT_3       3      ///< Acquisition frame count 3 of the UI setting
#define  CORRECTION_UI_FRAME_COUNT_MAX     4      ///< Maximum acquisition frame count

#define  CORRECTION_ACTUAL_FRAME_COUNT_1   1      ///< Flat field correction actual acquisition frame count 1
#define  CORRECTION_ACTUAL_FRAME_COUNT_2   2      ///< Flat field correction actual acquisition frame count 2
#define  CORRECTION_ACTUAL_FRAME_COUNT_4   4      ///< Flat field correction actual acquisition frame count 4
#define  CORRECTION_ACTUAL_FRAME_COUNT_8   8      ///< Flat field correction actual acquisition frame count 8
#define  CORRECTION_ACTUAL_FRAME_COUNT_16  16     ///< Flat field correction actual acquisition frame count 16

#define  IMAGE_DATA_NUM_MAX                16      ///< Maximum image data number
#define  IMAGE_DATA_NUM_MIN                8       ///< Minimum image data number

#define  GX_VERIFY(emStatus) \
	                        if (emStatus != GX_STATUS_SUCCESS)\
							{\
							ShowErrorString(emStatus); \
							return ;\
							 } 

#define VERIFY_STATUS_RET(emStatus) \
	                               if (emStatus != GX_STATUS_SUCCESS) \
								   {\
								   return emStatus;\
								   }\

/// File error code
typedef enum  FFC_STATUS
{
	FFC_STATUS_SUCCESS          =  0,   ///< Process file successfully
	FFC_STATUS_FILE_OPEN_ERROR  = -1,   ///< Open file error
	FFC_STATUS_FILE_WRITE_ERROR = -2,   ///< Write file error
	FFC_STATUS_FILE_READ_ERROR  = -3,   ///< Read file error
	FFC_STATUS_FILE_PATH_ERROR  = -4,   ///< File path error
	FFC_STATUS_FILE_DATA_ERROR  = -5    ///< File data error
} FFC_STATUS;

/////////////////////////////////////////////////////////////////////////////
// CGxFlatFieldCorrectionDlg dialog
#include "GxIAPI.h"
#include "DxImageProc.h"
#include <string>
#include <iostream>
#include <new>

class CGxFlatFieldCorrectionDlg : public CDialog
{
// Construction
public:
	CGxFlatFieldCorrectionDlg(CWnd* pParent = NULL);           ///< Standard constructor
	
private:
	GX_DEV_HANDLE       m_hDevice;                             ///< The handle for device

	BITMAPINFO          *m_pBmpInfo;	                       ///< The pointer for showing image
	BYTE                *m_pRawBuffer;                         ///< The pointer for the raw data of image
	BYTE                *m_pImgBuffer;	                       ///< The pointer for showing bmp image
    char                m_chBmpBuf[2048];	                   ///< The buffer for bmp image

	int64_t             m_nPayLoadSize;                        ///< Image payload size
	int64_t             m_nImageHeight;                        ///< Image height
    int64_t             m_nImageWidth;                         ///< Image width
	int64_t             m_nPixelColorFilter;                   ///< PixelColorFilter
	int64_t             m_nAutoWhiteBalance;                   ///< BalanceWhiteAuto
	int64_t             m_nRatioSelector;                      ///< BalanceWhiteRatioSelector

	GX_FLOAT_RANGE      m_stShutterFloatRange;                 ///< The range of ExposureTime
	GX_FLOAT_RANGE      m_stGainFloatRange;                    ///< The range of Gain
	GX_FLOAT_RANGE      m_stFloatRange;                        ///< Save the range of BalanceRatio  

	bool                m_bDevOpened;                          ///< The flag indicates whether the first device has been opened
	bool                m_bIsSnap;                             ///< The flag indicates whether image acquisition has started
 
	bool                m_bColorFilter;                        ///< The flag indicates whether the device supports ColorFilter 
	bool                m_bBalanceRatioSelector;               ///< The flag indicates whether the device supports BalanceRatioSelector

	CWnd                *m_pWnd;                               ///< The pointer for showing image window
	HDC                 m_hDC;                                 ///< The handle for drawing image DC

	BYTE                *m_pBrightBuf;                         ///< The pointer for the data of bright field image
	BYTE                *m_pDarkBuf;                           ///< The pointer for the data of dark field image
	bool                m_bDarkImgAcquired;                    ///< The flag indicates whether to acquire dark field images
	bool                m_bDarkImgAcquiredOnce;                ///< The flag indicates whether the dark field image has been acquired
	bool                m_bBrightImgAcquired;                  ///< The flag indicates whether to acquire bright field images
	bool                m_bBrightImgAcquiredOnce;              ///< The flag indicates whether the bright field image has been acquired
	DX_ACTUAL_BITS      m_nActualBits;                         ///< Image actual bits
	
	bool                m_bGetFFCCoefficients;                 ///< The flag indicates whether the coefficients of flat field correction has been calculated successfully
	bool                m_bExecuteFFC;                         ///< The flag indicates whether flat field correction has been executed
	bool                m_bIsFFC;                              ///< The flag indicates whether flat field correction is to be performed on the acquired image
	bool                m_bShowFFCOK;                          ///< The flag indicates whether the information indicating the successful execution of the flat field correction has been displayed
	
	BYTE                *m_pFFCCoefficients;                   ///< The pointer for the coefficients of flat field correction
	int                 m_nFFClength;                          ///< Flat field correction(FFC) coefficients length(byte)
	int                 m_nFFCGray;                            ///< FFC expected gray value
	int                 m_nNewGray;                            ///< Gray value of the UI setting
	bool                m_bFFCGray;                            ///< The flag indicates whether the set gray value is to be used
	int                 m_nFrameCount;                         ///< Acquisition Frame Count
	int                 m_nBrightImgNum;                       ///< Acquisition Number of bright field images
	int                 m_nAverageGrayDarkImg;                 ///< Average gray value of dark field image
	int                 m_nAverageGrayBrightImg;               ///< Average gray value of bright field image
	VxUint32            *m_pAccumulateImgBuf;                  ///< The pointer for the data of accumulated bright field image
	BYTE                *m_pAverageImgBuf;                     ///< The pointer for the data of average bright field image

	FLAT_FIELD_CORRECTION_PROCESS    m_stFlatFieldCorrection;  ///< Flat field correction process struct
public:

	bool PrepareForShowImg();

	bool PrepareForShowColorImg();

	bool PrepareForShowMonoImg();

	void InitUI();

	void UpDateUI();

	static void __stdcall OnFrameCallbackFun(GX_FRAME_CALLBACK_PARAM* pFrame);

	void UnPrepareForShowImg();

	void DrawImg();

	void ShowErrorString(GX_STATUS emErrorStatus);

	void InitEnumUI(GX_FEATURE_ID_CMD emFeatureID, CComboBox *pComboBox, bool bIsImplement);

	void InitWhiteRatioUI();

	void RefreshWhiteRatio();

	void InitGainUI();

	void InitShutterUI();

	void InitGrayUI();

	void InitFrameCountUI();

	GX_STATUS InitDevice();

	GX_STATUS SetPixelFormat8bit();

	GX_STATUS GetDeviceParam();

	DX_STATUS AccumulateImage(void *pImgBuf, int nDataWidth, int64_t nlength, VxUint32 *pAccumulateImgBuf);

	DX_STATUS GetAverageImage(VxUint32 *pImgBuf, int64_t nlength, int nAccumulateNum, int nDataWidth, void *pAverageImgBuf);

	int64_t GetAverageGray8bit(BYTE *pImgBuf, int64_t nImgSize);

	void AcquireDarkFieldImg(void);

	void AcquireBrightFieldImg(void);

	void GetFrameCount(int &nFrameCount);

	FFC_STATUS ReadFFCCoefficientsFile(const std::string &strFilePath);

	FFC_STATUS WriteFFCCoefficientsFile(const std::string &strFilePath);

// Dialog Data
	//{{AFX_DATA(CGxFlatFieldCorrectionDlg)
	enum { IDD = IDD_GxFlatFieldCorrection_DIALOG };
	double      m_dBalanceRatio;
	double	    m_dGainValue;
	double	    m_dShutterValue;
	CSliderCtrl m_sliderFrameCount;
	CSliderCtrl m_sliderGrayValue;
	//}}AFX_DATA

	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CGxFlatFieldCorrectionDlg)
	public:
	virtual BOOL PreTranslateMessage(MSG* pMsg);
	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support
	//}}AFX_VIRTUAL

// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	//{{AFX_MSG(CGxFlatFieldCorrectionDlg)
	virtual BOOL OnInitDialog();
	afx_msg HCURSOR OnQueryDragIcon();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg void OnClose();
	afx_msg void OnCbnSelchangeComboWriteBlance();
	afx_msg void OnBnClickedBtnOpenDevice();
	afx_msg void OnBnClickedBtnCloseDevice();
	afx_msg void OnBnClickedBtnStopSnap();
	afx_msg void OnBnClickedBtnStartSnap();
	afx_msg void OnCbnSelchangeComboRatioSelector();
	afx_msg void OnEnKillfocusEditBalanceRatio();
	afx_msg void OnKillfocusEditShutter();
	afx_msg void OnKillfocusEditGain();
	afx_msg void OnKillfocusEditGray();
	afx_msg void OnTimer(UINT_PTR nIDEvent);
	afx_msg void OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar);
	afx_msg void OnDeltaposSpinExpectGrayValue(NMHDR *pNMHDR, LRESULT *pResult);

	afx_msg void OnBnClickedButtonAcquireDarkImg();
	afx_msg void OnBnClickedButtonAcquireBrightImg();
	afx_msg void OnBnClickedButtonExecuteFFC();         
	afx_msg void OnBnClickedCheckPreview();             
	afx_msg void OnBnClickedCheckGray();               
	afx_msg void OnBnClickedButtonRead();
	afx_msg void OnBnClickedButtonWrite();
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()

};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_GxFlatFieldCorrectionDLG_H__D8D963C2_3BE2_44EC_B0EB_FCE016DDD6FA__INCLUDED_)
