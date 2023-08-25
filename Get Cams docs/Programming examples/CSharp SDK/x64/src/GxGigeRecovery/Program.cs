using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using GxIAPINET;

namespace GxGigeRecovery
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                // Call the CGigeRecovery class for demonstrating how to recovery from the offLine status 
                CGigeRecovery objCtrDevice = new CGigeRecovery();
                objCtrDevice.GigeRecovery();

                // Press any key to exit the console application
                Console.Write("\n<press any key to exit>\n");
                while (!System.Console.KeyAvailable) ;
                IGXFactory.GetInstance().Uninit();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
        }
    }
}
