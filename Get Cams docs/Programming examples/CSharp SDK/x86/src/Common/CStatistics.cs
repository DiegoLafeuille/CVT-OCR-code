using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace GxIAPINET.Sample.Common
{
    public class CStatistics
    {
        double m_dSumTime = 0.0;      ///< Total time
        double m_dMax     = 0.0;      ///< Max value
        double m_dMin     = 0.0;      ///< Min value
        Int64 m_nNum = 0;             ///< Statistics time count

        /// Construction
        public CStatistics()
        {
            m_dSumTime = 0.0;
            m_dMax = 0.0;
            m_dMin = 0.0;
            m_nNum = 0;
        }

        /// <summary>
        /// Increase the statistical count and calculate the relevant data
        /// </summary>
        /// <param name="dData"></param>
        public void AddStatisticalData(double dData)
        {
            __IncSumTimeData(dData);
            __UpdateData(dData);
        }

        /// <summary>
        /// Get Max
        /// </summary>
        /// <returns></returns>
        public double GetMax()
        {
            return m_dMax;
        }

        /// <summary>
        /// Get Min
        /// </summary>
        /// <returns></returns>
        public double GetMin()
        {
            return m_dMin;
        }

        /// <summary>
        /// Get average
        /// </summary>
        /// <returns></returns>
        public double GetAverage()
        {
            if (m_nNum == 0)
            {
                return 0;
            }

            return m_dSumTime / m_nNum;
        }

        /// <summary>
        /// Reset data
        /// </summary>
        public void Reset()
        {
            m_dSumTime = 0.0;
            m_dMax = 0.0;
            m_dMin = 0.0;
            m_nNum = 0;
        }

        /// <summary>
        /// Increase the total count of time
        /// </summary>
        /// <param name="dData"></param>
        private void __IncSumTimeData(double dData)
        {
            m_dSumTime = m_dSumTime + dData;
            m_nNum++;
        }

        /// <summary>
        /// Compare data and Update Max and Min
        /// </summary>
        /// <param name="dData"></param>
        private void __UpdateData(double dData)
        {
            // When the number of statistics is 1
            if (m_nNum == 1)
            {
                m_dMax = dData;
                m_dMin = dData;
                return;
            }

            if (dData > m_dMax)
            {
                m_dMax = dData;
            }

            if (dData < m_dMin)
            {
                m_dMin = dData;
            }
        }
    }
}
