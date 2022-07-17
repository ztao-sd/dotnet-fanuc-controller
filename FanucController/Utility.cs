using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;
using System.Globalization;
using System.Diagnostics;
using Serilog;
using Serilog.Core;
using Serilog.Events;
using Serilog.Configuration;
using CsvHelper;
using CsvHelper.Configuration;
using CsvHelper.Configuration.Attributes;
using MathNet.Numerics.LinearAlgebra;
using Python.Runtime;

namespace FanucController
{

    #region Data IO

    public static class Csv
    {

        private static CsvConfiguration appendConfig = new CsvConfiguration(CultureInfo.InvariantCulture)
        {
            HasHeaderRecord = false,
        };

        private static CsvConfiguration readConfig = new CsvConfiguration(CultureInfo.InvariantCulture)
        {
            HeaderValidated = null,
            MissingFieldFound = null,
        };

        public static void WriteCsv<T>(string path, List<T> dataList, bool append = false)
        {
            using (var writer = new StreamWriter(path, append: append))
            {
                if (!append)
                {
                    using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture))
                    {
                        csv.WriteRecords(dataList);
                    }
                }
                else
                {
                    using (var csv = new CsvWriter(writer, appendConfig))
                    {
                        csv.WriteRecords(dataList);
                    }
                }

            }
        }

        public static void AppendCsv<T>(string path, T data, bool append = true, bool header = false)
        {
            using (var writer = new StreamWriter(path, append: append))
            {
                if (header)
                {
                    using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture))
                    {
                        csv.WriteHeader<T>();
                        csv.NextRecord();
                    }
                }
                else
                {
                    using (var csv = new CsvWriter(writer, appendConfig))
                    {
                        csv.WriteRecord(data);
                        csv.NextRecord();
                    }
                }
            }
        }

        public static List<T> ReadCsv<T>(string path)
        {
            List<T> records = new List<T>();
            using (var reader = new StreamReader(path))
            {
                using (var csv = new CsvReader(reader, readConfig))
                {
                    records = csv.GetRecords<T>().ToList(); 
                }
            }
            return records;
        }

        public static List<PoseData> ReadPoseDataCsv(string path)
        {
            List<PoseData> records = new List<PoseData>();
            using (var reader = new StreamReader(path))
            {
                using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture))
                {
                    csv.Context.RegisterClassMap<PoseDataMap>();
                    records = csv.GetRecords<PoseData>().ToList();
                }
            }
            return records;
        }
    }

    public class PoseDataMap : ClassMap<PoseData>
    {
        public PoseDataMap()
        {
            Map(m => m.X).Name("x");
            Map(m => m.Y).Name("y");
            Map(m => m.Z).Name("z");
            Map(m => m.Gamma).Name("gamma");
            Map(m => m.Beta).Name("beta");
            Map(m => m.Alpha).Name("alpha");
        }
    }

    public class PoseData
    {
        [Name("time")] public double Time { get; set; }
        [Name("x")] public double X { get; set; }
        [Name("y")] public double Y { get; set; }
        [Name("z")] public double Z { get; set; }
        [Name("gamma")] public double Gamma { get; set; }
        [Name("beta")] public double Beta { get; set; }
        [Name("alpha")] public double Alpha { get; set; }
        
        

        public PoseData()
        {
            // Only here for CsvHelper to read file properly
        }

        public PoseData(Vector<double> pose, double time)
        {
            Time = time;
            X = pose[0]; Y = pose[1]; Z = pose[2];
            Alpha = pose[5]; Beta = pose[4]; Gamma = pose[3];
        }

        public PoseData(double[] pose, double time)
        {
            Time = time;
            X = pose[0]; Y = pose[1]; Z = pose[2];
            Alpha = pose[5]; Beta = pose[4]; Gamma = pose[3];
        }

        public Vector<double> ToVector()
        {
            var vector = CreateVector.Dense<double>(6);
            vector[0] = X; vector[1] = Y; vector[2] = Z;
            vector[5] = Alpha; vector[4] = Beta; vector[3] = Gamma;
            return vector;
        }
    }

    public class PidCoefficients
    {

        [Name("time")] public double Time { get; set; }
        [Name("KpX")] public double KpX { get; set; }
        [Name("KpY")] public double KpY { get; set; }
        [Name("KpZ")] public double KpZ { get; set; }
        [Name("KiX")] public double KiX { get; set; }
        [Name("KiY")] public double KiY { get; set; }
        [Name("KiZ")] public double KiZ { get; set; }
        [Name("KdX")] public double KdX { get; set; }
        [Name("KdY")] public double KdY { get; set; }
        [Name("KdZ")] public double KdZ { get; set; }
        

        public PidCoefficients()
        {

        }

        public PidCoefficients(double[] kValues, double time)
        {
            Time = time;
            KpX = kValues[0];
            KpY = kValues[1];
            KpZ = kValues[2];
            KiX = kValues[3];
            KiY = kValues[4];
            KiZ = kValues[5];
            KdX = kValues[6];
            KdY = kValues[7];
            KdZ = kValues[8];
        }

    }

    public class PidCoefficients6D
    {

        [Name("time")] public double Time { get; set; }
        [Name("KpX")] public double KpX { get; set; }
        [Name("KpY")] public double KpY { get; set; }
        [Name("KpZ")] public double KpZ { get; set; }
        [Name("KpW")] public double KpW { get; set; }
        [Name("KpP")] public double KpP { get; set; }
        [Name("KpR")] public double KpR { get; set; }
        [Name("KiX")] public double KiX { get; set; }
        [Name("KiY")] public double KiY { get; set; }
        [Name("KiZ")] public double KiZ { get; set; }
        [Name("KiW")] public double KiW { get; set; }
        [Name("KiP")] public double KiP { get; set; }
        [Name("KiR")] public double KiR { get; set; }
        [Name("KdX")] public double KdX { get; set; }
        [Name("KdY")] public double KdY { get; set; }
        [Name("KdZ")] public double KdZ { get; set; }
        [Name("KdW")] public double KdW { get; set; }
        [Name("KdP")] public double KdP { get; set; }
        [Name("KdR")] public double KdR { get; set; }


        public PidCoefficients6D()
        {

        }

        public PidCoefficients6D(double[] kValues, double time)
        {
            Time = time;
            KpX = kValues[0];
            KpY = kValues[1];
            KpZ = kValues[2];
            KpW = kValues[3];
            KpP = kValues[4];
            KpR = kValues[5];
            KiX = kValues[6];
            KiY = kValues[7];
            KiZ = kValues[8];
            KiW = kValues[9];
            KiP = kValues[10];
            KiR = kValues[11];
            KdX = kValues[12];
            KdY = kValues[13];
            KdZ = kValues[14];
            KdW = kValues[15];
            KdP = kValues[16];
            KdR = kValues[17];
        }

    }

    public class JointData
    {
        [Name("time")] public double Time { get; set; }
        [Name("j1")] public double J1 { get; set; }
        [Name("j2")] public double J2 { get; set; }
        [Name("j3")] public double J3 { get; set; }
        [Name("j4")] public double J4 { get; set; }
        [Name("j5")] public double J5 { get; set; }
        [Name("j6")] public double J6 { get; set; }

        public JointData(Vector<double> pose, double time)
        {
            Time = time;
            J1 = pose[0]; J2 = pose[1]; J3 = pose[2];
            J4 = pose[3]; J5 = pose[4]; J6 = pose[5];
        }

        public JointData(double[] pose, double time)
        {
            Time = time;
            J1 = pose[0]; J2 = pose[1]; J3 = pose[2];
            J4 = pose[3]; J5 = pose[4]; J6 = pose[5];
        }
    }

    public class Buffer<T>
    {
        public int Index;
        public int Size;
        public bool Full;
        private T[] memory;

        public Buffer(int size)
        {
            Size = size;
            Index = 0;
            memory = new T[size];
            Full = false;

        }

        public T[] Memory
        {
            get
            {
                if (Full)
                {
                    return memory;
                }
                else
                {

                    var subArray = new T[Index];
                    Array.Copy(memory, 0, subArray, 0, Index);
                    return subArray;
                }
            }
        }

        public void Reset()
        {
            Array.Clear(memory, 0, memory.Length);
            Index = 0;
            Full = false;
        }

        public void Add(T item)
        {
            memory[this.Index] = item;
            this.Index++;
            if (this.Index == this.Size)
            {
                this.Full = true;
                this.Index = 0;
            }
        }
    }

    #endregion

    #region Logger

    public class LogDisplay
    {
        private static LogDisplay localInstance;
        private ListView listView;
        private List<ListViewItem> listViewItems;
        private int size;
        private int index;
        private bool full;

        public LogDisplay(ListView listView, int bufferSize)
        {
            this.listView = listView;
            this.listViewItems = new List<ListViewItem>();
            this.size = bufferSize;
            this.index = 0;
            this.full = false;
        }

        public static void Initiate(ListView listView, int bufferSize)
        {
            if (localInstance == null)
            {
                localInstance = new LogDisplay(listView, bufferSize);
            }
        }

        public static LogDisplay Instance { get { return localInstance; } }

        public static void WriteLog(LogEvent logEvent)
        {
            if (localInstance != null)
            {
                ListViewItem listViewItem = new ListViewItem(logEvent.Level.ToString());
                listViewItem.SubItems.Add(logEvent.RenderMessage());
                localInstance.AddItem(listViewItem);
            }
        }

        public IList<ListViewItem> Items => listViewItems.AsReadOnly();

        private void AddItem(ListViewItem listViewItem, bool limit = false)
        {
            this.listViewItems.Add(listViewItem);
            if (!this.full || !limit)
            {
                this.listView.Items.Insert(this.index, listViewItem);
                this.index++;
            }
            else
            {
                ListViewItem tempItem;
                for (int i = 1; i < this.size; i++)
                {
                    tempItem = this.listView.Items[i];
                    this.listView.Items.RemoveAt(i);
                    this.listView.Items.Insert(i - 1, tempItem);
                }
                this.listView.Items.RemoveAt(this.size - 1);
                this.listView.Items.Insert(this.size - 1, listViewItem);
            }
            if (this.index == this.size) { this.full = true; }
        }
    }

    public class MemorySink : ILogEventSink
    {
        public event EventHandler<LogEvent> LogEvent;
        private static MemorySink localInstance;
        private readonly List<LogEvent> logEvents;
        public object memoryLock = new object();


        public MemorySink()
        {
            logEvents = new List<LogEvent>();
        }

        public static MemorySink Instance
        {
            get
            {
                if (localInstance == null)
                {
                    localInstance = new MemorySink();
                }
                return localInstance;
            }
        }

        public void Emit(LogEvent logEvent)
        {
            lock (memoryLock)
            {
                logEvents.Add(logEvent);
                LogDisplay.WriteLog(logEvent);
            }
        }

        // Return a object implementing IList
        public IList<LogEvent> LogEvents => logEvents.AsReadOnly();

        public void Reset()
        {
            logEvents.Clear();
        }
    }

    public static class MemorySinkExtensions
    {
        const string DefaultOutputTemplate = "[{Timestamp:HH:mm:ss} {Level:u3}] {Message:lj}{NewLine}{Exception}";

        public static LoggerConfiguration Memory(
            this LoggerSinkConfiguration sinkConfiguration,
            LogEventLevel restrictedToMinimumLevel = LevelAlias.Minimum,
            string outputTemplate = DefaultOutputTemplate,
            LoggingLevelSwitch levelSwitch = null)
        {
            return sinkConfiguration.Sink(MemorySink.Instance, restrictedToMinimumLevel, levelSwitch);
        }
    }

    #endregion

    #region Python

    public static class PythonScripts
    {
        private const string scriptDir = @"D:\LocalRepos\dotnet-fanuc-controller\PythonNeuralNetPControl";
        private const string pythonEnvDir = @"C:\Users\admin\anaconda3\envs\fanuc_rl";
        private const string pythonEnvPath = @"C:\Users\admin\anaconda3\envs\fanuc_rl\python.exe";

        public static void Run(string scriptName, string[] args = null, bool shell=false, string dir=null)
        {
            string scriptPath;
            if (dir is null)
            {
                scriptPath = Path.Combine(scriptDir, scriptName);
            }
            else
            {
                scriptPath = Path.Combine(dir, scriptName);
            }

            // Process configuration
            ProcessStartInfo startInfo = new ProcessStartInfo();
            startInfo.FileName = pythonEnvPath;
            if (shell == false)
            {
                startInfo.UseShellExecute = false;
                startInfo.CreateNoWindow = true;
                startInfo.RedirectStandardOutput = true;
                startInfo.RedirectStandardError = true;
            }
            else
            {
                startInfo.UseShellExecute = true;
                startInfo.CreateNoWindow = false;
                startInfo.RedirectStandardOutput = false;
                startInfo.RedirectStandardError = false;
            }
            
            if (args != null)
            {
                startInfo.Arguments = $"\"{scriptPath}\" \"{String.Join("\" \"", args)}\"";
            }
            else
            {
                startInfo.Arguments = $"\"{scriptPath}\"";
            }

            // Executing the process
            using (Process process = Process.Start(startInfo))
            {
                if (shell == false)
                {
                    using (StreamReader reader = process.StandardOutput)
                    {
                        string stderr = process.StandardError.ReadToEnd(); // 
                        string result = reader.ReadToEnd();
                        Console.WriteLine(result);
                    }
                }
                process.WaitForExit();
            }
        }

        public static void RunParallel(string scriptName, string[] args = null, bool shell=false, string dir=null)
        {
            ThreadPool.QueueUserWorkItem((obj) => Run(scriptName, args, shell, dir));
            //ThreadStart threadDelegate = new ThreadStart(() => Run(scriptName, args, shell));
            //Thread thread = new Thread(threadDelegate);
            //thread.IsBackground = true;
            //thread.Start();
        }

    }

    #endregion
}
