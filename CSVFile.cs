using System;
using System.IO;

namespace BigBrian {
    public class CSVFile {

        private StreamWriter writer;

        public CSVFile(string header) {
            InitFile();
            Log(header);
        }

        ~CSVFile() {
            writer.Close();
        }

        public void Log(string entry) {
            writer.WriteLine(entry);
            writer.Flush();
        }

        public void InitFile() {
            var fs = File.Create($"logs\\{String.Format("{0:u}", DateTime.Now).Replace(' ', '_').Replace(':', '-')}.csv");
            writer = new StreamWriter(fs);
        }
    }
}