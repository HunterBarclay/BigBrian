using System;
using System.IO;

namespace BigBrian {
    public class CSVFile {

        private StreamWriter writer;

        public CSVFile(string name, string header) {
            InitFile(name);
            Log(header);
        }

        ~CSVFile() {
            writer.Close();
        }

        public void Log(string entry) {
            writer.WriteLine(entry);
            writer.Flush();
        }

        public void InitFile(string name) {
            string path = $"logs\\{name}_data.csv";
            // string path = $"logs\\{name}_{String.Format("{0:u}", DateTime.Now).Replace(' ', '_').Replace(':', '-')}.csv";
            var fs = File.Create(path);
            writer = new StreamWriter(fs);
        }
    }
}