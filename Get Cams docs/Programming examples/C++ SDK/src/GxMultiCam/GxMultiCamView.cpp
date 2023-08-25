// GxMuitCamView.cpp : implementation of the  CGxMuitCamView  class
//

#include "stdafx.h"
#include "GxMultiCam.h"

#include "GxMultiCamDoc.h"
#include "GxMultiCamView.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CGxMuitCamView

IMPLEMENT_DYNCREATE(CGxMuitCamView, CView)

BEGIN_MESSAGE_MAP(CGxMuitCamView, CView)
	// Standard printing commands
	ON_COMMAND(ID_FILE_PRINT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_DIRECT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_PREVIEW, &CView::OnFilePrintPreview)
END_MESSAGE_MAP()

// CGxMuitCamView construction/destruction

CGxMuitCamView::CGxMuitCamView()
{
	// TODO: add construction code here

}

CGxMuitCamView::~CGxMuitCamView()
{
}

BOOL CGxMuitCamView::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: Modify the Window class or styles here by modifying
	//  CREATESTRUCT cs 

	return CView::PreCreateWindow(cs);
}

// CGxMuitCamView drawing

void CGxMuitCamView::OnDraw(CDC* /*pDC*/)
{
	CGxMuitCamDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	if (!pDoc)
		return;

	// TODO: add draw code for native data here
}


// CGxMuitCamView printing

BOOL CGxMuitCamView::OnPreparePrinting(CPrintInfo* pInfo)
{
	// default preparation
	return DoPreparePrinting(pInfo);
}

void CGxMuitCamView::OnBeginPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: add extra initialization before printing
}

void CGxMuitCamView::OnEndPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: add cleanup after printing
}


// CGxMuitCamView diagnostics

#ifdef _DEBUG
void CGxMuitCamView::AssertValid() const
{
	CView::AssertValid();
}

void CGxMuitCamView::Dump(CDumpContext& dc) const
{
	CView::Dump(dc);
}

CGxMuitCamDoc* CGxMuitCamView::GetDocument() const // non-debug version is inline
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CGxMuitCamDoc)));
	return (CGxMuitCamDoc*)m_pDocument;
}
#endif //_DEBUG


// CGxMuitCamView message handlers