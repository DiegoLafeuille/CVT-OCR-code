// GxMuitCamView.h : interface of the CGxMuitCamView class
//


#pragma once


class CGxMuitCamView : public CView
{
protected: // create from serialization only
	CGxMuitCamView();
	DECLARE_DYNCREATE(CGxMuitCamView)

// Attributes
public:
	CGxMuitCamDoc* GetDocument() const;

// Operations
public:

// Overrides
public:
	virtual void OnDraw(CDC* pDC);  // overridden to draw this view
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
protected:
	virtual BOOL OnPreparePrinting(CPrintInfo* pInfo);
	virtual void OnBeginPrinting(CDC* pDC, CPrintInfo* pInfo);
	virtual void OnEndPrinting(CDC* pDC, CPrintInfo* pInfo);

// Implementation
public:
	virtual ~CGxMuitCamView();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// Generated message map functions
protected:
	DECLARE_MESSAGE_MAP()
};

#ifndef _DEBUG  // debug version in GxMuitCamView.cpp
inline CGxMuitCamDoc* CGxMuitCamView::GetDocument() const
   { return reinterpret_cast<CGxMuitCamDoc*>(m_pDocument); }
#endif

