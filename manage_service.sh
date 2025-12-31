#!/bin/bash

# FPL Streamlit Service Management Script
SERVICE_NAME="fpl-streamlit.service"

echo "🎯 FPL Streamlit Service Manager"
echo "==============================="

case "$1" in
    start)
        echo "🚀 Starting FPL Streamlit service..."
        sudo systemctl start $SERVICE_NAME
        sleep 2
        sudo systemctl status $SERVICE_NAME --no-pager
        ;;
    stop)
        echo "🛑 Stopping FPL Streamlit service..."
        sudo systemctl stop $SERVICE_NAME
        ;;
    restart)
        echo "🔄 Restarting FPL Streamlit service..."
        sudo systemctl restart $SERVICE_NAME
        sleep 2
        sudo systemctl status $SERVICE_NAME --no-pager
        ;;
    status)
        echo "📊 Service status:"
        sudo systemctl status $SERVICE_NAME --no-pager
        ;;
    logs)
        echo "📋 Recent logs:"
        sudo journalctl -u $SERVICE_NAME -n 20 --no-pager
        ;;
    enable)
        echo "✅ Enabling service to start on boot..."
        sudo systemctl enable $SERVICE_NAME
        ;;
    disable)
        echo "❌ Disabling service auto-start..."
        sudo systemctl disable $SERVICE_NAME
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|enable|disable}"
        echo ""
        echo "Commands:"
        echo "  start   - Start the service"
        echo "  stop    - Stop the service"
        echo "  restart - Restart the service"
        echo "  status  - Show service status"
        echo "  logs    - Show recent service logs"
        echo "  enable  - Enable auto-start on boot"
        echo "  disable - Disable auto-start on boot"
        echo ""
        echo "Service Details:"
        echo "  Name: $SERVICE_NAME"
        echo "  Port: 8502"
        echo "  URL: http://your-server-ip:8502"
        ;;
esac
