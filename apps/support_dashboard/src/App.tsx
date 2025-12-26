import React, { useState, useEffect } from 'react';
import {
  ThemeProvider,
  createTheme,
  CssBaseline,
  AppBar,
  Toolbar,
  Typography,
  Tabs,
  Tab,
  Box,
  Container,
  Grid,
  Card,
  CardContent,
  Alert,
  Snackbar,
  Chip,
  LinearProgress,
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import {
  Dashboard as DashboardIcon,
  Analytics as AnalyticsIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  TrendingUp as TrendingUpIcon,
} from '@mui/icons-material';

// Theme configuration
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    h4: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 500,
    },
  },
});

// Mock data for charts
const systemHealthData = [
  { time: '00:00', cpu: 45, memory: 60, disk: 30 },
  { time: '04:00', cpu: 52, memory: 65, disk: 32 },
  { time: '08:00', cpu: 78, memory: 80, disk: 35 },
  { time: '12:00', cpu: 85, memory: 85, disk: 38 },
  { time: '16:00', cpu: 72, memory: 75, disk: 40 },
  { time: '20:00', cpu: 58, memory: 68, disk: 42 },
];

const apiRequestsData = [
  { endpoint: '/api/v1/analysis', requests: 1250, errors: 12 },
  { endpoint: '/api/v1/patterns', requests: 890, errors: 8 },
  { endpoint: '/api/v1/backtest', requests: 650, errors: 15 },
  { endpoint: '/api/v1/historical', requests: 1100, errors: 5 },
  { endpoint: '/api/v1/tools', requests: 780, errors: 3 },
];

const errorDistributionData = [
  { name: 'Database Errors', value: 25, color: '#ff6b6b' },
  { name: 'API Errors', value: 35, color: '#4ecdc4' },
  { name: 'Network Errors', value: 20, color: '#45b7d1' },
  { name: 'Validation Errors', value: 20, color: '#f9ca24' },
];

interface AlertData {
  id: number;
  type: 'error' | 'warning' | 'info' | 'success';
  title: string;
  message: string;
  timestamp: Date;
}

const DashboardTab: React.FC = () => {
  const [alerts, setAlerts] = useState<AlertData[]>([
    {
      id: 1,
      type: 'error',
      title: 'Database Connection Issue',
      message: 'PostgreSQL connection pool exhausted',
      timestamp: new Date(),
    },
    {
      id: 2,
      type: 'warning',
      title: 'High Memory Usage',
      message: 'Memory usage above 80% threshold',
      timestamp: new Date(Date.now() - 300000),
    },
    {
      id: 3,
      type: 'success',
      title: 'Data Pipeline Completed',
      message: 'Daily data ingestion completed successfully',
      timestamp: new Date(Date.now() - 600000),
    },
  ]);

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        System Overview
      </Typography>

      {/* Key Metrics */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                System Health
              </Typography>
              <Typography variant="h5" component="div" color="success.main">
                <CheckCircleIcon sx={{ mr: 1 }} />
                98.5%
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Active Users
              </Typography>
              <Typography variant="h5" component="div">
                1,247
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                API Requests (24h)
              </Typography>
              <Typography variant="h5" component="div">
                45,892
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Error Rate
              </Typography>
              <Typography variant="h5" component="div" color="error.main">
                0.8%
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* System Health Chart */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            System Resources (Last 24 Hours)
          </Typography>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={systemHealthData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="cpu" stroke="#8884d8" name="CPU %" />
              <Line type="monotone" dataKey="memory" stroke="#82ca9d" name="Memory %" />
              <Line type="monotone" dataKey="disk" stroke="#ffc658" name="Disk %" />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Recent Alerts */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Recent Alerts
          </Typography>
          {alerts.map((alert) => (
            <Alert
              key={alert.id}
              severity={alert.type}
              sx={{ mb: 2 }}
              iconMapping={{
                error: <ErrorIcon />,
                warning: <WarningIcon />,
                info: <CheckCircleIcon />,
                success: <CheckCircleIcon />,
              }}
            >
              <Typography variant="subtitle2">{alert.title}</Typography>
              <Typography variant="body2">{alert.message}</Typography>
              <Typography variant="caption" color="textSecondary">
                {alert.timestamp.toLocaleTimeString()}
              </Typography>
            </Alert>
          ))}
        </CardContent>
      </Card>
    </Box>
  );
};

const AnalyticsTab: React.FC = () => {
  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        API Analytics
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                API Requests by Endpoint
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={apiRequestsData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="endpoint" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="requests" fill="#8884d8" name="Requests" />
                  <Bar dataKey="errors" fill="#ff6b6b" name="Errors" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Error Distribution
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={errorDistributionData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {errorDistributionData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Performance Metrics */}
      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Performance Metrics
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} md={3}>
              <Typography variant="body2" color="textSecondary">
                Average Response Time
              </Typography>
              <Typography variant="h6">245ms</Typography>
              <LinearProgress variant="determinate" value={75} sx={{ mt: 1 }} />
            </Grid>
            <Grid item xs={12} md={3}>
              <Typography variant="body2" color="textSecondary">
                Throughput
              </Typography>
              <Typography variant="h6">1,892 req/min</Typography>
              <LinearProgress variant="determinate" value={85} sx={{ mt: 1 }} />
            </Grid>
            <Grid item xs={12} md={3}>
              <Typography variant="body2" color="textSecondary">
                Cache Hit Rate
              </Typography>
              <Typography variant="h6">94.2%</Typography>
              <LinearProgress variant="determinate" value={94} sx={{ mt: 1 }} />
            </Grid>
            <Grid item xs={12} md={3}>
              <Typography variant="body2" color="textSecondary">
                Uptime
              </Typography>
              <Typography variant="h6">99.98%</Typography>
              <LinearProgress variant="determinate" value={99} sx={{ mt: 1 }} />
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </Box>
  );
};

const MonitoringTab: React.FC = () => {
  const [alerts, setAlerts] = useState<AlertData[]>([
    {
      id: 1,
      type: 'error',
      title: 'Critical: Database Down',
      message: 'Main PostgreSQL instance is unresponsive',
      timestamp: new Date(),
    },
    {
      id: 2,
      type: 'warning',
      title: 'High Latency Detected',
      message: 'API response time > 2s for 5 minutes',
      timestamp: new Date(Date.now() - 180000),
    },
    {
      id: 3,
      type: 'info',
      title: 'Scheduled Maintenance',
      message: 'Database backup starting in 30 minutes',
      timestamp: new Date(Date.now() - 900000),
    },
  ]);

  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');

  const handleAlertClick = (alert: AlertData) => {
    setSnackbarMessage(`Alert: ${alert.title} - ${alert.message}`);
    setSnackbarOpen(true);
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Real-time Monitoring
      </Typography>

      {/* Status Overview */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Database Status
              </Typography>
              <Chip
                label="CRITICAL"
                color="error"
                icon={<ErrorIcon />}
                sx={{ mb: 2 }}
              />
              <Typography variant="body2" color="textSecondary">
                Last checked: 2 minutes ago
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                API Gateway
              </Typography>
              <Chip
                label="HEALTHY"
                color="success"
                icon={<CheckCircleIcon />}
                sx={{ mb: 2 }}
              />
              <Typography variant="body2" color="textSecondary">
                Response time: 145ms
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Data Pipeline
              </Typography>
              <Chip
                label="WARNING"
                color="warning"
                icon={<WarningIcon />}
                sx={{ mb: 2 }}
              />
              <Typography variant="body2" color="textSecondary">
                Queue size: 1,247 items
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Active Alerts */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Active Alerts
          </Typography>
          {alerts.map((alert) => (
            <Alert
              key={alert.id}
              severity={alert.type}
              sx={{ mb: 2, cursor: 'pointer' }}
              onClick={() => handleAlertClick(alert)}
              iconMapping={{
                error: <ErrorIcon />,
                warning: <WarningIcon />,
                info: <CheckCircleIcon />,
                success: <CheckCircleIcon />,
              }}
            >
              <Typography variant="subtitle2">{alert.title}</Typography>
              <Typography variant="body2">{alert.message}</Typography>
              <Typography variant="caption" color="textSecondary">
                {alert.timestamp.toLocaleString()}
              </Typography>
            </Alert>
          ))}
        </CardContent>
      </Card>

      {/* Real-time Metrics */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Real-time Metrics
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Typography variant="body2" color="textSecondary">
                Current CPU Usage
              </Typography>
              <Typography variant="h6">67%</Typography>
              <LinearProgress variant="determinate" value={67} sx={{ mt: 1 }} />
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="body2" color="textSecondary">
                Memory Usage
              </Typography>
              <Typography variant="h6">82%</Typography>
              <LinearProgress variant="determinate" value={82} sx={{ mt: 1 }} />
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="body2" color="textSecondary">
                Active Connections
              </Typography>
              <Typography variant="h6">1,456</Typography>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="body2" color="textSecondary">
                Queue Depth
              </Typography>
              <Typography variant="h6">234</Typography>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      <Snackbar
        open={snackbarOpen}
        autoHideDuration={6000}
        onClose={() => setSnackbarOpen(false)}
        message={snackbarMessage}
      />
    </Box>
  );
};

const App: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AppBar position="static">
        <Toolbar>
          <DashboardIcon sx={{ mr: 2 }} />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Gravity Tech Support Dashboard
          </Typography>
          <Chip
            label="LIVE"
            color="success"
            size="small"
            icon={<CheckCircleIcon />}
          />
        </Toolbar>
      </AppBar>

      <Container maxWidth="xl">
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabValue} onChange={handleTabChange} aria-label="dashboard tabs">
            <Tab
              label="Dashboard"
              icon={<DashboardIcon />}
              iconPosition="start"
            />
            <Tab
              label="Analytics"
              icon={<AnalyticsIcon />}
              iconPosition="start"
            />
            <Tab
              label="Monitoring"
              icon={<WarningIcon />}
              iconPosition="start"
            />
          </Tabs>
        </Box>

        {tabValue === 0 && <DashboardTab />}
        {tabValue === 1 && <AnalyticsTab />}
        {tabValue === 2 && <MonitoringTab />}
      </Container>
    </ThemeProvider>
  );
};

export default App;
