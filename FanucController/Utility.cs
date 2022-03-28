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
            List<T> record;
            using (var reader = new StreamReader(path))
            {

                using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture))
                {
                    record = csv.GetRecords<T>().ToList();
                }
            }
            return record;
        }
    }

    public class PoseData
    {
        [Name("time")] public double Time { get; set; }
        [Name("x")] public double X { get; set; }
        [Name("y")] public double Y { get; set; }
        [Name("z")] public double Z { get; set; }
        [Name("alpha")] public double Alpha { get; set; }
        [Name("beta")] public double Beta { get; set; }
        [Name("gamma")] public double Gamma { get; set; }

        public PoseData(Vector<double> pose, double time)
        {
            Time = time;
            X = pose[0]; Y = pose[1]; Z = pose[2];
            Alpha = pose[3]; Beta = pose[4]; Gamma = pose[5];
        }

        public PoseData(double[] pose, double time)
        {
            Time = time;
            X = pose[0]; Y = pose[1]; Z = pose[2];
            Alpha = pose[3]; Beta = pose[4]; Gamma = pose[5];
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

        public static void Run(string scriptName, string[] args = null)
        {
            string scriptPath = Path.Combine(scriptDir, scriptName);

            // Process configuration
            ProcessStartInfo startInfo = new ProcessStartInfo();
            startInfo.FileName = pythonEnvPath;
            startInfo.UseShellExecute = false;
            startInfo.CreateNoWindow = true;
            startInfo.RedirectStandardOutput = true;
            startInfo.RedirectStandardError = true;
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
                using (StreamReader reader = process.StandardOutput)
                {
                    string stderr = process.StandardError.ReadToEnd(); // 
                    string result = reader.ReadToEnd();
                    Console.WriteLine(result);
                }
            }
        }

        public static void RunParallel(string scriptName, string[] args = null)
        {
            //ThreadPool.QueueUserWorkItem((obj) => Run(scriptName, args));
            ThreadStart threadDelegate = new ThreadStart(() => Run(scriptName, args));
            Thread thread = new Thread(threadDelegate);
            thread.IsBackground = true;
            thread.Start();
        }

    }

    #endregion
}
