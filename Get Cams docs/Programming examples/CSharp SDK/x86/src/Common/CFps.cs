using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;

namespace GxIAPINET.Sample.Common
{
   public class CFps
    {
        UInt64 m_nFrameCount = 0;                                                     ///< The number of frames accumulated since the last calculation
        double m_dBeginTime = 0.0;                                                    ///< A frame of time before the first frame(Initial 0)
        double m_dEndTime = 0.0;                                                      ///< Last frame time
        double m_dFps = 0.0;                                                          ///< The last frame rate(frame/second)
        double                m_dCurrentFps            = 0.0;                         ///< Current frame rate
        UInt64 m_nTotalFrameCount = 0;                                                ///< The cumulative number of frames
        CStopWatch            m_objTime                = new CStopWatch();            ///< timer
        object                m_objLock                = new object();

        /// <summary>
        /// Construction
        /// </summary>
        public CFps()
        {
            //Reset all parameters
            Reset();
        }


        /// <summary>
        ///  Get the latest frame rate
        /// </summary>
        /// <returns>current frame</returns>
        public double GetFps()
        {
            lock (m_objLock)
            {
                
                return m_dCurrentFps;
            }
        }

        /// <summary>
        /// Gets the total number of frames accumulated
        /// </summary>
        /// <returns>current frame</returns>
        public UInt64 GetTotalFrameCount()
        {
            lock (m_objLock)
            {
                return m_nTotalFrameCount;
            }
        }

        /// <summary>
        /// Increase the number of frames
        /// </summary>
        public void IncreaseFrameNum()
        {
            lock (m_objLock)
            {
                //Increase total the number of frames
                m_nTotalFrameCount++;

                //Increase the number of frames
                m_nFrameCount++;

                //The update interval
                m_dEndTime = m_objTime.ElapsedTime();
            }
        }

        /// <summary>
        /// Update frame rate
        /// If the function is called at a frequency that exceeds the frame rate, the frame rate drops to zero
        /// </summary>
        public void UpdateFps()
        {
            lock (m_objLock)
            {
                //Calculate the time interval
                double dInterval = m_dEndTime - m_dBeginTime;

                //Check whether the time interval equals zero
                if (dInterval > 0)
                {
                    m_dFps = 1000.0 * m_nFrameCount / dInterval;
                    m_nFrameCount = 0;              //The accumulated frames are cleared
                    m_dBeginTime = m_dEndTime;      //Update start time

                    m_dCurrentFps = m_dFps;
                }
                else if (dInterval == 0) //The time interval equals zero(no new frames)
                {
                    
                    if (m_dCurrentFps != 0)
                    {
                       
                        double dCurrentInterval = m_objTime.ElapsedTime() - m_dBeginTime;

                       
                        double dPeriod = 1000.0 / m_dCurrentFps;  
                        const double RATIO = 1.5;                      
                        double dThresh = RATIO * dPeriod;          

                        
                        const double ZERO_FPS_INTERVAL = 2000;
                        if (dCurrentInterval > ZERO_FPS_INTERVAL)
                        {
                            m_dCurrentFps = 0;
                        }
                       
                        else if (dCurrentInterval > dThresh)
                        {
                            m_dCurrentFps = m_dFps / (dCurrentInterval / (1000.0 / m_dFps));
                        }
                        else { }
                    }
                    else { }
                }
                else { }
            }

        }

        /// <summary>
        /// Restore the timer to an initial state
        /// </summary>
        public void Reset()
        {
            m_nFrameCount = 0;
            m_dBeginTime = 0.0;
            m_dEndTime = 0.0;
            m_nTotalFrameCount = 0;
            m_dFps = 0.0;
            m_dCurrentFps = 0.0;
            m_objTime.Start();          //restart timer
        }
    }
}
