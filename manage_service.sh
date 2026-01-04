#!/bin/bash

# FPL FastAPI/React Service Management Script
BACKEND_SERVICE="fpl-backend.service"
FRONTEND_SERVICE="fpl-frontend.service"

echo "🎯 FPL FastAPI/React Service Manager"
echo "==================================="

# Function to manage services
manage_service() {
    local service_name=$1
    local action=$2

    case "$action" in
        start)
            echo "🚀 Starting $service_name..."
            sudo systemctl start $service_name
            sleep 2
            sudo systemctl status $service_name --no-pager
            ;;
        stop)
            echo "🛑 Stopping $service_name..."
            sudo systemctl stop $service_name
            ;;
        restart)
            echo "🔄 Restarting $service_name..."
            sudo systemctl restart $service_name
            sleep 2
            sudo systemctl status $service_name --no-pager
            ;;
        status)
            echo "📊 $service_name status:"
            sudo systemctl status $service_name --no-pager
            ;;
        logs)
            echo "📋 $service_name recent logs:"
            sudo journalctl -u $service_name -n 20 --no-pager
            ;;
        enable)
            echo "✅ Enabling $service_name to start on boot..."
            sudo systemctl enable $service_name
            ;;
        disable)
            echo "❌ Disabling $service_name auto-start..."
            sudo systemctl disable $service_name
            ;;
    esac
}

case "$1" in
    backend)
        case "$2" in
            start|stop|restart|status|logs|enable|disable)
                manage_service $BACKEND_SERVICE $2
                ;;
            *)
                echo "Usage: $0 backend {start|stop|restart|status|logs|enable|disable}"
                ;;
        esac
        ;;
    frontend)
        case "$2" in
            start|stop|restart|status|logs|enable|disable)
                manage_service $FRONTEND_SERVICE $2
                ;;
            *)
                echo "Usage: $0 frontend {start|stop|restart|status|logs|enable|disable}"
                ;;
        esac
        ;;
    start)
        echo "🚀 Starting all FPL services..."
        manage_service $BACKEND_SERVICE start
        echo ""
        manage_service $FRONTEND_SERVICE start
        ;;
    stop)
        echo "🛑 Stopping all FPL services..."
        manage_service $FRONTEND_SERVICE stop
        echo ""
        manage_service $BACKEND_SERVICE stop
        ;;
    restart)
        echo "🔄 Restarting all FPL services..."
        manage_service $BACKEND_SERVICE restart
        echo ""
        manage_service $FRONTEND_SERVICE restart
        ;;
    status)
        echo "📊 All FPL services status:"
        manage_service $BACKEND_SERVICE status
        echo ""
        manage_service $FRONTEND_SERVICE status
        ;;
    logs)
        echo "📋 All FPL services recent logs:"
        echo "=== BACKEND LOGS ==="
        manage_service $BACKEND_SERVICE logs
        echo ""
        echo "=== FRONTEND LOGS ==="
        manage_service $FRONTEND_SERVICE logs
        ;;
    enable)
        echo "✅ Enabling all services to start on boot..."
        manage_service $BACKEND_SERVICE enable
        manage_service $FRONTEND_SERVICE enable
        ;;
    disable)
        echo "❌ Disabling all services auto-start..."
        manage_service $BACKEND_SERVICE disable
        manage_service $FRONTEND_SERVICE disable
        ;;
    *)
        echo "Usage: $0 {backend|frontend} {start|stop|restart|status|logs|enable|disable}"
        echo "   or: $0 {start|stop|restart|status|logs|enable|disable}  # for all services"
        echo ""
        echo "Commands:"
        echo "  backend  - Manage FastAPI backend service only"
        echo "  frontend - Manage React frontend service only"
        echo "  start    - Start all services"
        echo "  stop     - Stop all services"
        echo "  restart  - Restart all services"
        echo "  status   - Show all services status"
        echo "  logs     - Show all services recent logs"
        echo "  enable   - Enable auto-start on boot for all services"
        echo "  disable  - Disable auto-start on boot for all services"
        echo ""
        echo "Service Details:"
        echo "  Backend:  $BACKEND_SERVICE  (Port 8000) - http://your-server-ip:8000"
        echo "  Frontend: $FRONTEND_SERVICE (Port 5173) - http://your-server-ip:5173"
        echo "  API Docs: http://your-server-ip:8000/docs"
        ;;
esac
