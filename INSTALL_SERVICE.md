# Installing Concept Explorer as a System Service

To have Concept Explorer start automatically on boot and keep running:

## Option 1: Quick Manual Start (Current Method)
```bash
cd /home/richard/shared/ConceptExplorer
./start.sh
```

## Option 2: Install as Systemd Service (Recommended for Production)

1. Copy the service file:
```bash
sudo cp concept-explorer.service /etc/systemd/system/
```

2. Reload systemd:
```bash
sudo systemctl daemon-reload
```

3. Enable and start the service:
```bash
sudo systemctl enable concept-explorer
sudo systemctl start concept-explorer
```

4. Check status:
```bash
sudo systemctl status concept-explorer
```

## Managing the Service

```bash
# Start
sudo systemctl start concept-explorer

# Stop
sudo systemctl stop concept-explorer

# Restart
sudo systemctl restart concept-explorer

# View logs
sudo journalctl -u concept-explorer -f

# Disable auto-start
sudo systemctl disable concept-explorer
```

## Accessing the Service

- Local: http://localhost:5059
- Network: http://192.168.0.94:5059
- Dashboard: http://192.168.0.94:5001 (look for "Concept Explorer" in Development section)

## Troubleshooting

If the service fails to start:

1. Check if port 5059 is available:
```bash
lsof -i :5059
```

2. Test manual start:
```bash
cd /home/richard/shared/ConceptExplorer
source venv/bin/activate
python concept_explorer.py --port 5059
```

3. Check logs:
```bash
sudo journalctl -u concept-explorer -n 50
```
