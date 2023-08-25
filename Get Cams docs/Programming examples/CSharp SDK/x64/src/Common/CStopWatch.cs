using System;
using System.Diagnostics;

namespace GxIAPINET.Sample.Common
{

    public class CStopWatch
    {
        double m_dStartTime = 0.0;    ///< Start time
        double m_dStopTime = 0.0;     ///< Stop time


        public CStopWatch()
        {
            Start();
        }

        /// <summary>
        /// Start counting
        /// </summary>
        public void Start()
        {
            m_dStartTime = Stopwatch.GetTimestamp();
        }

        /// <summary>
        /// Stop counting
        /// </summary>
        /// <returns>Time difference unit ms</returns>
        public double Stop()
        {
            m_dStopTime = Stopwatch.GetTimestamp();
            double theElapsedTime = ElapsedTime();

            m_dStartTime = m_dStopTime;
            return theElapsedTime;
        }


        /// <summary>
        /// Get time difference
        /// </summary>
        /// <returns>Time difference unit ms</returns>
        public double ElapsedTime()
        {
            m_dStopTime = Stopwatch.GetTimestamp();
            double dTimeElapsed = (m_dStopTime - m_dStartTime) * 1000.0;
          
            return dTimeElapsed / Stopwatch.Frequency;
        }
    }
}
