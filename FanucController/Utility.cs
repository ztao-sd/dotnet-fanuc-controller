using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Serilog;
using Serilog.Core;
using Serilog.Events;
using Serilog.Configuration;

namespace FanucController
{

    public class Buffer<T>
    {
        public int Index;
        public int Size;
        public bool Full;
        private T[] memory;

        public Buffer(int size)
        {
            this.Index = 0;
            this.memory = new T[size];
            this.Full = false;
        }

        public T[] Memory
        {
            get
            {
                if (this.Full)
                {
                    return this.memory;
                }
                else
                {

                    var subArray = new T[this.Index];
                    Array.Copy(this.memory, 0, subArray, 0, this.Index);
                    return subArray;
                }
            }
        }

        public void Reset()
        {
            Array.Clear(this.memory, 0, this.memory.Length);
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
                localInstance= new LogDisplay(listView, bufferSize);
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

        private void AddItem(ListViewItem listViewItem, bool limit=false)
        {
            this.listViewItems.Add(listViewItem);
            if (!this.full || !limit) { 
                this.listView.Items.Insert(this.index, listViewItem);
                this.index++;
            }
            else {
                ListViewItem tempItem;
                for (int i = 1; i < this.size; i++) 
                {
                    tempItem = this.listView.Items[i];
                    this.listView.Items.RemoveAt(i);
                    this.listView.Items.Insert(i-1,tempItem);
                }
                this.listView.Items.RemoveAt(this.size-1);
                this.listView.Items.Insert(this.size-1, listViewItem);
            }
            if (this.index == this.size) { this.full = true; }
        }
    }

    public class MemorySink : ILogEventSink
    {
        public event EventHandler<LogEvent> LogEvent;
        private static MemorySink localInstance;
        private readonly List<LogEvent> logEvents;
       
    
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
            logEvents.Add(logEvent);
            LogDisplay.WriteLog(logEvent);
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

}
