import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import PitchView from '../components/PitchView';
import PlayersTable from '../components/PlayersTable';
import SoccerPitch from '../components/SoccerPitch';
import { getDreamTeam } from '../services/api';
import {
  Home,
  Users,
  Target,
  Settings,
  LogOut,
  Zap,
  User,
  ChevronLeft,
  ChevronRight,
  RefreshCw,
  AlertCircle,
  Loader2
} from 'lucide-react';

const Dashboard = () => {
  const { user, logout, isAuthenticated, loading } = useAuth();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [activeTab, setActiveTab] = useState('optimize');

  // Dream Team state
  const [dreamTeam, setDreamTeam] = useState([]);
  const [dreamTeamStats, setDreamTeamStats] = useState({ cost: 0, xp: 0 });
  const [dreamTeamLoading, setDreamTeamLoading] = useState(false);
  const [dreamTeamError, setDreamTeamError] = useState(null);

  const menuItems = [
    { id: 'overview', label: 'Overview', icon: Home },
    { id: 'players', label: 'Players', icon: Users },
    { id: 'optimize', label: 'Dream Team', icon: Target },
    { id: 'team', label: 'My Team', icon: User },
    { id: 'settings', label: 'Settings', icon: Settings },
  ];

  // Fetch dream team data on component mount
  useEffect(() => {
    fetchDreamTeam();
  }, []);

  const fetchDreamTeam = async () => {
    setDreamTeamLoading(true);
    setDreamTeamError(null);

    try {
      const response = await getDreamTeam();
      if (response.success && response.squad) {
        setDreamTeam(response.squad);
        setDreamTeamStats({
          cost: response.total_cost || 0,
          xp: response.total_xP || 0
        });
      } else {
        setDreamTeamError(response.message || 'Failed to fetch dream team');
      }
    } catch (error) {
      console.error('Error fetching dream team:', error);

      // Show the REAL API error message to user
      let errorMessage = 'Dream Team Error: ';

      if (error.response) {
        // Server responded with error status
        const status = error.response.status;
        const data = error.response.data;

        console.error('Server Error Response:', { status, data });

        if (data?.detail) {
          errorMessage += data.detail;
        } else if (data?.message) {
          errorMessage += data.message;
        } else if (status === 401) {
          errorMessage += 'Authentication Required - Please login first';
        } else if (status === 500) {
          errorMessage += 'Server Error - Please try again later';
        } else {
          errorMessage += `HTTP ${status} Error`;
        }
      } else if (error.request) {
        // Network error
        console.error('Network Error - No response received');
        errorMessage += 'Network Error - Cannot connect to server. Check if backend is running on port 8000.';
      } else {
        // Other error
        console.error('Unknown Error:', error.message);
        errorMessage += error.message || 'Unknown error occurred.';
      }

      setDreamTeamError(errorMessage);
    } finally {
      setDreamTeamLoading(false);
    }
  };

  const handleLogout = () => {
    logout();
  };

  // Show loading spinner while checking authentication
  if (loading) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-slate-400">Loading...</p>
        </div>
      </div>
    );
  }

  // Redirect to login if not authenticated
  if (!isAuthenticated()) {
    console.warn('🔐 [DASHBOARD] User not authenticated, redirecting to login');
    window.location.href = '/login';
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-2"></div>
          <p className="text-slate-400">Redirecting to login...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-900 flex">
      {/* Sidebar */}
      <div className={`bg-slate-800 border-r border-slate-700 transition-all duration-300 ${
        sidebarCollapsed ? 'w-16' : 'w-64'
      }`}>
        {/* Sidebar Header */}
        <div className="p-4 border-b border-slate-700">
          <div className="flex items-center justify-between">
            {!sidebarCollapsed && (
              <div className="flex items-center space-x-3">
                <div className="h-8 w-8 bg-blue-500 rounded-lg flex items-center justify-center">
                  <Zap className="h-5 w-5 text-white" />
                </div>
                <div>
                  <h1 className="text-lg font-bold text-white">FPL Quant Pro</h1>
                </div>
              </div>
            )}
            <button
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
              className="p-1 rounded-md hover:bg-slate-700 text-slate-400 hover:text-white transition-colors"
            >
              {sidebarCollapsed ? (
                <ChevronRight className="h-5 w-5" />
              ) : (
                <ChevronLeft className="h-5 w-5" />
              )}
            </button>
          </div>
        </div>

        {/* Navigation Menu */}
        <nav className="p-4 space-y-2">
          {menuItems.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`w-full flex items-center space-x-3 px-3 py-2 rounded-md transition-colors ${
                  activeTab === item.id
                    ? 'bg-blue-600 text-white'
                    : 'text-slate-300 hover:bg-slate-700 hover:text-white'
                }`}
              >
                <Icon className="h-5 w-5 flex-shrink-0" />
                {!sidebarCollapsed && (
                  <span className="text-sm font-medium">{item.label}</span>
                )}
              </button>
            );
          })}
        </nav>

        {/* Logout Button */}
        <div className="absolute bottom-4 left-4">
          <button
            onClick={handleLogout}
            className="inline-flex items-center gap-3 px-4 py-2 rounded-lg transition-colors hover:bg-slate-700 text-slate-300 hover:text-white"
          >
            <LogOut className="h-5 w-5 flex-shrink-0" />
            {!sidebarCollapsed && (
              <span className="text-sm font-medium">Logout</span>
            )}
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="bg-slate-800 border-b border-slate-700 px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-semibold text-white capitalize">
                {menuItems.find(item => item.id === activeTab)?.label || 'Dashboard'}
              </h2>
              <p className="text-slate-400 text-sm">
                Welcome back, {user?.username || 'User'}
              </p>
            </div>
            <div className="flex items-center space-x-4">
              {activeTab === 'optimize' && (
                <button
                  onClick={fetchDreamTeam}
                  disabled={dreamTeamLoading}
                  className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 disabled:opacity-50 text-white px-4 py-2 rounded-md transition-colors"
                >
                  {dreamTeamLoading ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <RefreshCw className="h-4 w-4" />
                  )}
                  <span>Recalculate</span>
                </button>
              )}
              <div className="h-8 w-8 bg-emerald-500 rounded-full flex items-center justify-center">
                <span className="text-white text-sm font-medium">
                  {user?.username?.charAt(0).toUpperCase() || 'U'}
                </span>
              </div>
            </div>
          </div>
        </header>

        {/* Content Area */}
        <main className="flex-1 p-6">
          <div className="max-w-7xl mx-auto">
            {/* Content based on active tab */}
            {activeTab === 'overview' && (
              <div className="space-y-6">
                {/* Stats Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                  <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-slate-400 text-sm">Total Points</p>
                        <p className="text-2xl font-bold text-white">2,450</p>
                      </div>
                      <div className="h-12 w-12 bg-blue-500 rounded-full flex items-center justify-center">
                        <Target className="h-6 w-6 text-white" />
                      </div>
                    </div>
                  </div>

                  <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-slate-400 text-sm">Rank</p>
                        <p className="text-2xl font-bold text-white">127,543</p>
                      </div>
                      <div className="h-12 w-12 bg-emerald-500 rounded-full flex items-center justify-center">
                        <Users className="h-6 w-6 text-white" />
                      </div>
                    </div>
                  </div>

                  <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-slate-400 text-sm">Gameweek</p>
                        <p className="text-2xl font-bold text-white">19</p>
                      </div>
                      <div className="h-12 w-12 bg-purple-500 rounded-full flex items-center justify-center">
                        <Zap className="h-6 w-6 text-white" />
                      </div>
                    </div>
                  </div>

                  <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-slate-400 text-sm">Bank Balance</p>
                        <p className="text-2xl font-bold text-white">£2.5M</p>
                      </div>
                      <div className="h-12 w-12 bg-yellow-500 rounded-full flex items-center justify-center">
                        <span className="text-white font-bold">£</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Welcome Message */}
                <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
                  <h3 className="text-lg font-semibold text-white mb-4">Welcome to FPL Quant Pro</h3>
                  <p className="text-slate-400 mb-4">
                    Your AI-powered FPL optimization platform is ready. Navigate through the sidebar to explore players,
                    optimize your dream team, and manage your squad with advanced analytics.
                  </p>
                  <div className="flex space-x-4">
                    <button className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md transition-colors">
                      Explore Players
                    </button>
                    <button className="bg-emerald-600 hover:bg-emerald-700 text-white px-4 py-2 rounded-md transition-colors">
                      Optimize Dream Team
                    </button>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'players' && (
              <PlayersTable />
            )}

            {activeTab === 'optimize' && (
              <div className="space-y-6">
                {/* Loading State */}
                {dreamTeamLoading && (
                  <div className="flex items-center justify-center py-12">
                    <div className="text-center">
                      <Loader2 className="h-8 w-8 animate-spin text-blue-500 mx-auto mb-4" />
                      <p className="text-slate-400">Optimizing your dream team...</p>
                    </div>
                  </div>
                )}

                {/* Error State */}
                {dreamTeamError && !dreamTeamLoading && (
                  <div className="bg-red-900/20 border border-red-500/50 rounded-lg p-6">
                    <div className="flex items-center space-x-3">
                      <AlertCircle className="h-6 w-6 text-red-400 flex-shrink-0" />
                      <div>
                        <h3 className="text-lg font-semibold text-red-400 mb-2">Optimization Failed</h3>
                        <p className="text-red-300">{dreamTeamError}</p>
                        <button
                          onClick={fetchDreamTeam}
                          className="mt-3 bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-md transition-colors"
                        >
                          Try Again
                        </button>
                      </div>
                    </div>
                  </div>
                )}

                {/* Success State - Show Dream Team */}
                {!dreamTeamLoading && !dreamTeamError && dreamTeam.length > 0 && (
                  <div className="space-y-6">
                    {/* Stats Cards */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                      <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-slate-400 text-sm">Total Players</p>
                            <p className="text-2xl font-bold text-white">{dreamTeam.length}</p>
                          </div>
                          <div className="h-12 w-12 bg-blue-500 rounded-full flex items-center justify-center">
                            <Users className="h-6 w-6 text-white" />
                          </div>
                        </div>
                      </div>

                      <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-slate-400 text-sm">Total Cost</p>
                            <p className="text-2xl font-bold text-emerald-400">£{dreamTeamStats.cost.toFixed(1)}M</p>
                          </div>
                          <div className="h-12 w-12 bg-emerald-500 rounded-full flex items-center justify-center">
                            <span className="text-white font-bold">£</span>
                          </div>
                        </div>
                      </div>

                      <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-slate-400 text-sm">Total xP</p>
                            <p className="text-2xl font-bold text-purple-400">{dreamTeamStats.xp.toFixed(1)}</p>
                          </div>
                          <div className="h-12 w-12 bg-purple-500 rounded-full flex items-center justify-center">
                            <Zap className="h-6 w-6 text-white" />
                          </div>
                        </div>
                      </div>

                      <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
                        <div className="flex items-center justify-between mb-4">
                          <div>
                            <p className="text-slate-400 text-sm">Budget Used</p>
                            <p className="text-2xl font-bold text-yellow-400">{((dreamTeamStats.cost / 100) * 100).toFixed(1)}%</p>
                          </div>
                          <div className="h-12 w-12 bg-yellow-500 rounded-full flex items-center justify-center">
                            <Target className="h-6 w-6 text-white" />
                          </div>
                        </div>
                        {/* Progress Bar */}
                        <div className="w-full bg-slate-700 rounded-full h-2">
                          <div
                            className={`h-2 rounded-full transition-all duration-500 ${
                              dreamTeamStats.cost <= 95 ? 'bg-green-500' :
                              dreamTeamStats.cost <= 100 ? 'bg-yellow-500' :
                              'bg-red-500'
                            }`}
                            style={{ width: `${Math.min(dreamTeamStats.cost, 150)}%` }}
                          ></div>
                        </div>
                        {/* Budget Status Text */}
                        <div className="mt-2 text-xs text-center">
                          <span className={`font-medium ${
                            dreamTeamStats.cost <= 95 ? 'text-green-400' :
                            dreamTeamStats.cost <= 100 ? 'text-yellow-400' :
                            'text-red-400'
                          }`}>
                            {dreamTeamStats.cost <= 95 ? 'Under Budget' :
                             dreamTeamStats.cost <= 100 ? 'On Budget' :
                             'Over Budget'}
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Dream Team Display - Only Pitch View */}
                    <SoccerPitch players={dreamTeam} />
                  </div>
                )}
                  <div className="space-y-6">
                    {/* Stats Cards */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                      <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-slate-400 text-sm">Total Players</p>
                            <p className="text-2xl font-bold text-white">{dreamTeam.length}</p>
                          </div>
                          <div className="h-12 w-12 bg-blue-500 rounded-full flex items-center justify-center">
                            <Users className="h-6 w-6 text-white" />
                          </div>
                        </div>
                      </div>

                      <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-slate-400 text-sm">Total Cost</p>
                            <p className="text-2xl font-bold text-emerald-400">£{dreamTeamStats.cost.toFixed(1)}M</p>
                          </div>
                          <div className="h-12 w-12 bg-emerald-500 rounded-full flex items-center justify-center">
                            <span className="text-white font-bold">£</span>
                          </div>
                        </div>
                      </div>

                      <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-slate-400 text-sm">Total xP</p>
                            <p className="text-2xl font-bold text-purple-400">{dreamTeamStats.xp.toFixed(1)}</p>
                          </div>
                          <div className="h-12 w-12 bg-purple-500 rounded-full flex items-center justify-center">
                            <Zap className="h-6 w-6 text-white" />
                          </div>
                        </div>
                      </div>

                      <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-slate-400 text-sm">Budget Used</p>
                            <p className="text-2xl font-bold text-yellow-400">{((dreamTeamStats.cost / 100) * 100).toFixed(1)}%</p>
                          </div>
                          <div className="h-12 w-12 bg-yellow-500 rounded-full flex items-center justify-center">
                            <Target className="h-6 w-6 text-white" />
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Dream Team Table */}
                    <div className="bg-slate-800 rounded-lg border border-slate-700 overflow-hidden">
                      <div className="px-6 py-4 border-b border-slate-700">
                        <h3 className="text-lg font-semibold text-white">Dream Team Squad</h3>
                        <p className="text-slate-400 text-sm">Optimized 15-player squad with maximum expected points</p>
                      </div>

                      <div className="overflow-x-auto">
                        <table className="w-full">
                          <thead className="bg-slate-700/50">
                            <tr>
                              <th className="px-6 py-3 text-left text-xs font-medium text-slate-300 uppercase tracking-wider">Position</th>
                              <th className="px-6 py-3 text-left text-xs font-medium text-slate-300 uppercase tracking-wider">Player</th>
                              <th className="px-6 py-3 text-left text-xs font-medium text-slate-300 uppercase tracking-wider">Team</th>
                              <th className="px-6 py-3 text-left text-xs font-medium text-slate-300 uppercase tracking-wider">Price</th>
                              <th className="px-6 py-3 text-left text-xs font-medium text-slate-300 uppercase tracking-wider">xP</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-700">
                            {dreamTeam.map((player, index) => (
                              <tr key={player.id || index} className="hover:bg-slate-700/30">
                                <td className="px-6 py-4 whitespace-nowrap">
                                  <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                    player.position === 'GKP' ? 'bg-yellow-100 text-yellow-800' :
                                    player.position === 'DEF' ? 'bg-blue-100 text-blue-800' :
                                    player.position === 'MID' ? 'bg-green-100 text-green-800' :
                                    'bg-red-100 text-red-800'
                                  }`}>
                                    {player.position}
                                  </span>
                                </td>
                                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-white">
                                  {player.name}
                                </td>
                                <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-300">
                                  {player.team_name}
                                </td>
                                <td className="px-6 py-4 whitespace-nowrap text-sm text-emerald-400">
                                  £{player.price.toFixed(1)}M
                                </td>
                                <td className="px-6 py-4 whitespace-nowrap text-sm text-purple-400">
                                  {player.predicted_xP?.toFixed(1) || '0.0'}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                

                {/* Empty State */}
                {!dreamTeamLoading && !dreamTeamError && dreamTeam.length === 0 && (
                  <div className="bg-slate-800 rounded-lg p-6 border border-slate-700 text-center">
                    <Target className="h-12 w-12 text-slate-400 mx-auto mb-4" />
                    <h3 className="text-lg font-semibold text-white mb-2">No Dream Team Available</h3>
                    <p className="text-slate-400 mb-4">
                      Your dream team will appear here once optimization is complete.
                    </p>
                    <button
                      onClick={fetchDreamTeam}
                      className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-md transition-colors"
                    >
                      Generate Dream Team
                    </button>
                  </div>
                )}
              </div>
            )}

            {activeTab === 'team' && (
              <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
                <h3 className="text-lg font-semibold text-white mb-4">My Team</h3>
                <p className="text-slate-400">Your current FPL team management will be available here.</p>
              </div>
            )}

            {activeTab === 'settings' && (
              <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
                <h3 className="text-lg font-semibold text-white mb-4">Settings</h3>
                <p className="text-slate-400">Application settings and preferences will be available here.</p>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
};

export default Dashboard;
