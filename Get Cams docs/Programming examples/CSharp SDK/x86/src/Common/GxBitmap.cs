using System;
using System.Runtime.InteropServices;
using System.Drawing;
using System.Drawing.Imaging;
using System.Windows.Forms;
using System.IO;
using GxIAPINET;

namespace GxIAPINET.Sample.Common
{
  public  class GxBitmap
    {

      IGXDevice      m_objIGXDevice    = null;                ///<The handle for device
      PictureBox m_pic_ShowImage = null;                      ///<Picture display control
      bool m_bIsColor = false;                                ///<The flag indicates whether the color camera 
      byte[] m_byMonoBuffer = null;                           ///<Black and white camera buffer
      byte[] m_byColorBuffer = null;                          ///<Color camera buffer
      byte[] m_byRawBuffer = null;                            ///<The buffer used to store raw graphs
      int m_nPayloadSize = 0;                                 ///<Image data size
      int            m_nWidth          = 0;                   ///<Image width
      int            m_nHeigh          = 0;                   ///<Image height
      Bitmap         m_bitmapForSave   = null;                ///<bitmap object
      const uint PIXEL_FORMATE_BIT = 0x00FF0000;              ///<Used to AND the current data format to get the current data bits
      const uint GX_PIXEL_8BIT = 0x00080000;                  ///<8-bit data image format
                                                 
      const int      COLORONCOLOR      = 3;
      const uint     DIB_RGB_COLORS    = 0;
      const uint     SRCCOPY           = 0x00CC0020;
      CWin32Bitmaps.BITMAPINFO m_objBitmapInfo = new CWin32Bitmaps.BITMAPINFO();
      IntPtr m_pBitmapInfo = IntPtr.Zero;
      Graphics m_objGC = null;
      IntPtr m_pHDC = IntPtr.Zero;
      
                                                       
      /// <summary>
      /// The construction function
      /// </summary>
      /// <param name="objIGXDevice">Device object</param>
      /// <param name="objPictureBox">Picture display control</param>
      public GxBitmap(IGXDevice objIGXDevice, PictureBox objPictureBox)
      {
          m_objIGXDevice = objIGXDevice;
          m_pic_ShowImage = objPictureBox;
          string strValue = null;
          if (null != objIGXDevice)
          {
              //Get image data
              m_nPayloadSize = (int)objIGXDevice.GetRemoteFeatureControl().GetIntFeature("PayloadSize").GetValue();
              m_nWidth = (int)objIGXDevice.GetRemoteFeatureControl().GetIntFeature("Width").GetValue();
              m_nHeigh = (int)objIGXDevice.GetRemoteFeatureControl().GetIntFeature("Height").GetValue();

              //Check whether the color camera
              if (objIGXDevice.GetRemoteFeatureControl().IsImplemented("PixelColorFilter"))
              {
                  strValue = objIGXDevice.GetRemoteFeatureControl().GetEnumFeature("PixelColorFilter").GetValue();

                  if ("None" != strValue)
                  {
                      m_bIsColor = true;
                  }
              }
          }

          //Apply buffer to cache image data
          m_byRawBuffer = new byte[m_nPayloadSize];
          m_byMonoBuffer = new byte[__GetStride(m_nWidth, m_bIsColor) * m_nHeigh];
          m_byColorBuffer = new byte[__GetStride(m_nWidth, m_bIsColor) * m_nHeigh];

          __CreateBitmap(out m_bitmapForSave, m_nWidth, m_nHeigh, m_bIsColor);

          m_objGC = m_pic_ShowImage.CreateGraphics();
          m_pHDC = m_objGC.GetHdc();
          if (m_bIsColor)
          {
              m_objBitmapInfo.bmiHeader.biSize = (uint)Marshal.SizeOf(typeof(CWin32Bitmaps.BITMAPINFOHEADER));
              m_objBitmapInfo.bmiHeader.biWidth = m_nWidth;
              m_objBitmapInfo.bmiHeader.biHeight = -m_nHeigh;
              m_objBitmapInfo.bmiHeader.biPlanes = 1;
              m_objBitmapInfo.bmiHeader.biBitCount = 24;
              m_objBitmapInfo.bmiHeader.biCompression = 0;
              m_objBitmapInfo.bmiHeader.biSizeImage = 0;
              m_objBitmapInfo.bmiHeader.biXPelsPerMeter = 0;
              m_objBitmapInfo.bmiHeader.biYPelsPerMeter = 0;
              m_objBitmapInfo.bmiHeader.biClrUsed = 0;
              m_objBitmapInfo.bmiHeader.biClrImportant = 0;
          }
          else
          {
              m_objBitmapInfo.bmiHeader.biSize = (uint)Marshal.SizeOf(typeof(CWin32Bitmaps.BITMAPINFOHEADER));
              m_objBitmapInfo.bmiHeader.biWidth = m_nWidth;
              m_objBitmapInfo.bmiHeader.biHeight = -m_nHeigh;
              m_objBitmapInfo.bmiHeader.biPlanes = 1;
              m_objBitmapInfo.bmiHeader.biBitCount = 8;
              m_objBitmapInfo.bmiHeader.biCompression = 0;
              m_objBitmapInfo.bmiHeader.biSizeImage = 0;
              m_objBitmapInfo.bmiHeader.biXPelsPerMeter = 0;
              m_objBitmapInfo.bmiHeader.biYPelsPerMeter = 0;
              m_objBitmapInfo.bmiHeader.biClrUsed = 0;
              m_objBitmapInfo.bmiHeader.biClrImportant = 0;

              m_objBitmapInfo.bmiColors = new CWin32Bitmaps.RGBQUAD[256];
              // black and white image need initialize palette
              for (int i = 0; i < 256; i++)
              {
                  m_objBitmapInfo.bmiColors[i].rgbBlue     = (byte)i;
                  m_objBitmapInfo.bmiColors[i].rgbGreen    = (byte)i;
                  m_objBitmapInfo.bmiColors[i].rgbRed      = (byte)i;
                  m_objBitmapInfo.bmiColors[i].rgbReserved = (byte)i;
              }
          }
          m_pBitmapInfo = Marshal.AllocHGlobal(2048);
          Marshal.StructureToPtr(m_objBitmapInfo, m_pBitmapInfo, false);
      }

      /// <summary>
      ///  Process and Display the image
      /// </summary>
      /// <param name="objCfg">Image processing adjustment parameter object</param>
      /// <param name="objIBaseData">Image data object</param>
      public void ShowImageProcess(IImageProcessConfig objCfg, IBaseData objIBaseData)
      {
          //Check whether update bitmap information
          __UpdateBufferSize(objIBaseData);

          if (null != objIBaseData)
          {
              if (GX_FRAME_STATUS_LIST.GX_FRAME_STATUS_SUCCESS == objIBaseData.GetStatus())
              {
                  if (m_bIsColor)
                  {
                      IntPtr pBufferColor = objIBaseData.ImageProcess(objCfg);
                      Marshal.Copy(pBufferColor, m_byColorBuffer, 0, __GetStride(m_nWidth ,m_bIsColor)* m_nHeigh);
                      __ShowImage(m_byColorBuffer);
                  }
                  else
                  {
                      IntPtr pBufferMono = objIBaseData.ImageProcess(objCfg);
                      Marshal.Copy(pBufferMono, m_byMonoBuffer, 0, __GetStride(m_nWidth, m_bIsColor) * m_nHeigh);

                      __ShowImage(m_byMonoBuffer);
                  }
              }
          }       
      }
      

      /// <summary>
      /// Display image
      /// </summary>
      /// <param name="objIBaseData">Image data object</param>
      public void Show(IBaseData objIBaseData)
      {
          GX_VALID_BIT_LIST emValidBits = GX_VALID_BIT_LIST.GX_BIT_0_7;

          //Check whether update bitmap information
          __UpdateBufferSize(objIBaseData);
       

          if (null != objIBaseData)
          {
              emValidBits = __GetBestValudBit(objIBaseData.GetPixelFormat());
              if (GX_FRAME_STATUS_LIST.GX_FRAME_STATUS_SUCCESS == objIBaseData.GetStatus())
              {
                  if (m_bIsColor)
                  {
                      IntPtr pBufferColor = objIBaseData.ConvertToRGB24(emValidBits, GX_BAYER_CONVERT_TYPE_LIST.GX_RAW2RGB_NEIGHBOUR, false);
                      Marshal.Copy(pBufferColor, m_byColorBuffer, 0, __GetStride(m_nWidth, m_bIsColor) * m_nHeigh);
                      __ShowImage(m_byColorBuffer);
                  }
                  else
                  {
                      IntPtr pBufferMono = IntPtr.Zero;
                      if (__IsPixelFormat8(objIBaseData.GetPixelFormat()))
                      {
                          pBufferMono = objIBaseData.GetBuffer();
                      }
                      else
                      {
                          pBufferMono = objIBaseData.ConvertToRaw8(emValidBits);
                      }
                      Marshal.Copy(pBufferMono, m_byMonoBuffer, 0, __GetStride(m_nWidth, m_bIsColor) * m_nHeigh);

                      __ShowImage(m_byMonoBuffer);
                  }
              }
          }
      }

      /// <summary>
      /// the image are stored as BMP format
      /// </summary>
      /// <param name="objIBaseData">Image data object</param>
      /// <param name="strFilePath">File name</param>
      public void SaveBmp(IBaseData objIBaseData, string strFilePath)
      {
          GX_VALID_BIT_LIST emValidBits = GX_VALID_BIT_LIST.GX_BIT_0_7;

          //Check whether update bitmap information
          __UpdateBufferSize(objIBaseData);

          if (null != objIBaseData)
          {
              emValidBits = __GetBestValudBit(objIBaseData.GetPixelFormat());
              if (m_bIsColor)
              {
                  IntPtr pBufferColor = objIBaseData.ConvertToRGB24(emValidBits, GX_BAYER_CONVERT_TYPE_LIST.GX_RAW2RGB_NEIGHBOUR, false);
                  Marshal.Copy(pBufferColor, m_byColorBuffer, 0, __GetStride(m_nWidth, m_bIsColor) * m_nHeigh);
                  __UpdateBitmapForSave(m_byColorBuffer);
              }
              else
              {
                  IntPtr pBufferMono = IntPtr.Zero;
                  if (__IsPixelFormat8(objIBaseData.GetPixelFormat()))
                  {
                      pBufferMono = objIBaseData.GetBuffer();
                  }
                  else
                  {
                      pBufferMono = objIBaseData.ConvertToRaw8(emValidBits);
                  }
                  Marshal.Copy(pBufferMono, m_byMonoBuffer, 0, __GetStride(m_nWidth, m_bIsColor) * m_nHeigh);

                  __UpdateBitmapForSave(m_byMonoBuffer);
              }
              m_bitmapForSave.Save(strFilePath, ImageFormat.Bmp);
          }
      }

      /// <summary>
      /// the image are stored as Raw format
      /// </summary>
      /// <param name="objIBaseData">Image data object</param>
      /// <param name="strFilePath">File name</param>
      public void SaveRaw(IBaseData objIBaseData, string strFilePath)
      {
          Stream objFileStream = new FileStream(strFilePath, FileMode.Create);
          BinaryWriter objSW = new BinaryWriter(objFileStream);

          //Check whether update bitmap information
          __UpdateBufferSize(objIBaseData);
          
          if (null != objIBaseData)
          {
              IntPtr pBufferRaw = objIBaseData.GetBuffer();
              Marshal.Copy(pBufferRaw, m_byRawBuffer, 0, m_nPayloadSize);
          }

          objSW.Write(m_byRawBuffer);
          objSW.Close();
          objFileStream.Close();
      }

      /// <summary>
      ///  //Check whether update bitmap information
      /// </summary>
      /// <param name="objIBaseData">Image data object</param>
      private void __UpdateBufferSize(IBaseData objIBaseData)
      {
          if (null != objIBaseData)
          {
              if (__IsCompatible(m_bitmapForSave, m_nWidth, m_nHeigh, m_bIsColor))
              {
                  m_nPayloadSize = (int)objIBaseData.GetPayloadSize();
                  m_nWidth = (int)objIBaseData.GetWidth();
                  m_nHeigh = (int)objIBaseData.GetHeight();
              }
              else
              {
                  m_nPayloadSize = (int)objIBaseData.GetPayloadSize();
                  m_nWidth = (int)objIBaseData.GetWidth();
                  m_nHeigh = (int)objIBaseData.GetHeight();

                  m_byRawBuffer = new byte[m_nPayloadSize];
                  m_byMonoBuffer = new byte[__GetStride(m_nWidth, m_bIsColor) * m_nHeigh];
                  m_byColorBuffer = new byte[__GetStride(m_nWidth, m_bIsColor) * m_nHeigh];

                  //Update BitmapInfo
                  m_objBitmapInfo.bmiHeader.biWidth = m_nWidth;
                  m_objBitmapInfo.bmiHeader.biHeight = m_nHeigh;
                  Marshal.StructureToPtr(m_objBitmapInfo, m_pBitmapInfo, false);
              }
          }
      }

      /// <summary>
      ///  Update Bitmap information
      /// </summary>
      /// <param name="byBuffer">Image buffer</param>
      private void __UpdateBitmapForSave(byte[] byBuffer)
      {
          if (__IsCompatible(m_bitmapForSave, m_nWidth, m_nHeigh, m_bIsColor))
          {
              __UpdateBitmap(m_bitmapForSave, byBuffer, m_nWidth, m_nHeigh, m_bIsColor);
          }
          else
          {
              __CreateBitmap(out m_bitmapForSave, m_nWidth, m_nHeigh, m_bIsColor);
              __UpdateBitmap(m_bitmapForSave, byBuffer, m_nWidth, m_nHeigh, m_bIsColor);
          }
      }

      /// <summary>
      /// Display image
      /// </summary>
      /// <param name="byBuffer">Image data buffer</param>
      private void __ShowImage(byte[] byBuffer)
      {
          if (null != m_pic_ShowImage)
          {
              CWin32Bitmaps.SetStretchBltMode(m_pHDC, COLORONCOLOR);
              CWin32Bitmaps.StretchDIBits(
                          m_pHDC,
                          0,
                          0,
                          m_pic_ShowImage.Width,
                          m_pic_ShowImage.Height,
                          0,
                          0,
                          m_nWidth,
                          m_nHeigh,
                          byBuffer,
                          m_pBitmapInfo,
                          DIB_RGB_COLORS,
                          SRCCOPY);            
          }
      }

      /// <summary>
      ///  Check whether the PixelFormat is 8 bit
      /// </summary>
      /// <param name="emPixelFormatEntry">Image data format</param>
      /// <returns>true:  is 8 bit,   false:  is not 8 bit</returns>
      private bool __IsPixelFormat8(GX_PIXEL_FORMAT_ENTRY emPixelFormatEntry)
      {
          bool bIsPixelFormat8 = false;
          uint uiPixelFormatEntry = (uint)emPixelFormatEntry;
          if ((uiPixelFormatEntry & PIXEL_FORMATE_BIT) == GX_PIXEL_8BIT)
          {
              bIsPixelFormat8 = true;
          }
          return bIsPixelFormat8;
      }
 
      /// <summary>
      /// Get the best Bit by GX_PIXEL_FORMAT_ENTRY
      /// </summary>
      /// <param name="em">Image format</param>
      /// <returns>the best Bit</returns>
      private GX_VALID_BIT_LIST __GetBestValudBit(GX_PIXEL_FORMAT_ENTRY emPixelFormatEntry)
      {
          GX_VALID_BIT_LIST emValidBits = GX_VALID_BIT_LIST.GX_BIT_0_7;
          switch (emPixelFormatEntry)
          {
              case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_MONO8:
              case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_BAYER_GR8:
              case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_BAYER_RG8:
              case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_BAYER_GB8:
              case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_BAYER_BG8:
                  {
                      emValidBits = GX_VALID_BIT_LIST.GX_BIT_0_7;
                      break;
                  }
              case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_MONO10:
              case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_BAYER_GR10:
              case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_BAYER_RG10:
              case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_BAYER_GB10:
              case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_BAYER_BG10:
                  {
                      emValidBits = GX_VALID_BIT_LIST.GX_BIT_2_9;
                      break;
                  }
              case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_MONO12:
              case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_BAYER_GR12:
              case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_BAYER_RG12:
              case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_BAYER_GB12:
              case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_BAYER_BG12:
                  {
                      emValidBits = GX_VALID_BIT_LIST.GX_BIT_4_11;
                      break;
                  }
              case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_MONO14:
                  {
                      //There is no such data format
                      break;
                  }
              case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_MONO16:
              case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_BAYER_GR16:
              case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_BAYER_RG16:
              case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_BAYER_GB16:
              case GX_PIXEL_FORMAT_ENTRY.GX_PIXEL_FORMAT_BAYER_BG16:
                  {
                      //There is no such data format
                      break;
                  }
              default:
                  break;
          }
          return emValidBits;
      }

      /// <summary>
      /// Get  the display image format
      /// </summary>
      /// <param name="bIsColor">The color camera</param>
      /// <returns>Image data format</returns>
      private PixelFormat __GetFormat(bool bIsColor)
      {
          return bIsColor ? PixelFormat.Format24bppRgb : PixelFormat.Format8bppIndexed;
      }

      /// <summary>
      /// Calculate the number of bytes occupied by width
      /// </summary>
      /// <param name="nWidth">Image width</param>
      /// <param name="bIsColor">Whether is a color camera</param>
      /// <returns>The number of bytes of an image line</returns>
      private int __GetStride(int nWidth, bool bIsColor)
      {
          return bIsColor ? nWidth * 3 : nWidth;
      }

      /// <summary>
      ///  whether compatible
      /// </summary>
      /// <param name="bitmap">Bitmap object</param>
      /// <param name="nWidth">Image width</param>
      /// <param name="nHeight">Image height</param>
      /// <param name="bIsColor">Whether is a color camera</param>
      /// <returns>true: compatible , false: not compatible</returns>
      private bool __IsCompatible(Bitmap bitmap, int nWidth, int nHeight, bool bIsColor)
      {
          if (bitmap == null
              || bitmap.Height != nHeight
              || bitmap.Width != nWidth
              || bitmap.PixelFormat != __GetFormat(bIsColor)
           )
          {
              return false;
          }
          return true;
      }

      /// <summary>
      /// Create Bitmap
      /// </summary>
      /// <param name="bitmap">Bitmap object</param>
      /// <param name="nWidth">Image width</param>
      /// <param name="nHeight">Image height</param>
      /// <param name="bIsColor">Whether is a color camera</param>
      private void __CreateBitmap(out Bitmap bitmap, int nWidth, int nHeight, bool bIsColor)
      {
          bitmap = new Bitmap(nWidth, nHeight, __GetFormat(bIsColor));
          if (bitmap.PixelFormat == PixelFormat.Format8bppIndexed)
          {
              ColorPalette colorPalette = bitmap.Palette;
              for (int i = 0; i < 256; i ++)
              {
                  colorPalette.Entries[i] = Color.FromArgb(i, i, i);
              }
              bitmap.Palette = colorPalette;
          }
      }

      /// <summary>
      /// Update and copy image data to Bitmap buffer
      /// </summary>
      /// <param name="bitmap">Bitmap object</param>
      /// <param name="nWidth">Image width</param>
      /// <param name="nHeight">Image height</param>
      /// <param name="bIsColor">Whether is a color camera</param>
      private void __UpdateBitmap(Bitmap bitmap, byte[] byBuffer, int nWidth, int nHeight, bool bIsColor)
      {
          //Lock the bitmap data
          BitmapData bmpData = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height), ImageLockMode.ReadWrite, bitmap.PixelFormat);

          //Get a buffer pointer to the Bitmap
          IntPtr ptrBmp = bmpData.Scan0;
          int nImageStride = __GetStride(m_nWidth, bIsColor);
          //The image width can be divisible by 4 and direct copy
          if (nImageStride == bmpData.Stride)
          {
              Marshal.Copy(byBuffer, 0, ptrBmp, bmpData.Stride * bitmap.Height);
          }
          else
          {
              for (int i = 0; i < bitmap.Height; ++i)
              {
                  Marshal.Copy(byBuffer, i * nImageStride, new IntPtr(ptrBmp.ToInt64() + i * bmpData.Stride), m_nWidth);
              }
          }
          //Unlock the bitmap data
          bitmap.UnlockBits(bmpData);
      }
    }
}
