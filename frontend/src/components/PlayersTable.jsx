import React, { useState, useEffect, useMemo } from 'react';
import { getPlayers } from '../services/api';
import {
  Search,
  ChevronUp,
  ChevronDown,
  Loader2,
  AlertCircle,
  Users,
  Filter
} from 'lucide-react';

const PlayersTable = () => {
  const [players, setPlayers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortConfig, setSortConfig] = useState({ key: 'predicted_xP', direction: 'desc' });
  const [currentPage, setCurrentPage] = useState(1);
  const [positionFilter, setPositionFilter] = useState('all');

  const itemsPerPage = 20;

  // Fetch players data
  useEffect(() => {
    fetchPlayers();
  }, []);

  const fetchPlayers = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await getPlayers();
      if (response.success && response.players) {
        setPlayers(response.players);
      } else {
        setError(response.message || 'Failed to fetch players data');
      }
    } catch (error) {
      console.error('Error fetching players:', error);
      let errorMessage = 'Failed to load players data';

      if (error.response) {
        const status = error.response.status;
        const data = error.response.data;

        if (data?.detail) {
          errorMessage = data.detail;
        } else if (status === 500) {
          errorMessage = 'Server error - please try again later';
        } else {
          errorMessage = `HTTP ${status} Error`;
        }
      } else if (error.request) {
        errorMessage = 'Cannot connect to server. Check if backend is running.';
      }

      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  // Sort players
  const handleSort = (key) => {
    let direction = 'asc';
    if (sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc';
    }
    setSortConfig({ key, direction });
  };

  // Filter and sort players
  const filteredAndSortedPlayers = useMemo(() => {
    let filtered = players;

    // Apply position filter
    if (positionFilter !== 'all') {
      filtered = filtered.filter(player => player.position === positionFilter);
    }

    // Apply search filter
    if (searchTerm) {
      filtered = filtered.filter(player =>
        player.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        player.team_name.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // Apply sorting
    filtered.sort((a, b) => {
      const aValue = a[sortConfig.key] || 0;
      const bValue = b[sortConfig.key] || 0;

      if (sortConfig.direction === 'asc') {
        return aValue > bValue ? 1 : -1;
      }
      return aValue < bValue ? 1 : -1;
    });

    return filtered;
  }, [players, searchTerm, sortConfig, positionFilter]);

  // Pagination
  const totalPages = Math.ceil(filteredAndSortedPlayers.length / itemsPerPage);
  const paginatedPlayers = filteredAndSortedPlayers.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  const resetFilters = () => {
    setSearchTerm('');
    setPositionFilter('all');
    setCurrentPage(1);
    setSortConfig({ key: 'predicted_xP', direction: 'desc' });
  };

  const getPositionColor = (position) => {
    switch (position) {
      case 'Goalkeeper': return 'bg-yellow-500';
      case 'Defender': return 'bg-blue-500';
      case 'Midfielder': return 'bg-green-500';
      case 'Forward': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getSortIcon = (columnKey) => {
    if (sortConfig.key !== columnKey) {
      return <ChevronUp className="h-4 w-4 opacity-30" />;
    }
    return sortConfig.direction === 'asc'
      ? <ChevronUp className="h-4 w-4" />
      : <ChevronDown className="h-4 w-4" />;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin text-blue-500 mx-auto mb-4" />
          <p className="text-slate-400">Loading players data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-900/20 border border-red-500/50 rounded-lg p-6">
        <div className="flex items-center space-x-3">
          <AlertCircle className="h-6 w-6 text-red-400 flex-shrink-0" />
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-red-400 mb-2">Failed to Load Players</h3>
            <p className="text-red-300 mb-4">{error}</p>
            <button
              onClick={fetchPlayers}
              className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-md transition-colors"
            >
              Try Again
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header and Filters */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div className="flex items-center space-x-2">
          <Users className="h-6 w-6 text-blue-500" />
          <h3 className="text-xl font-semibold text-white">
            Players Database
          </h3>
          <span className="text-sm text-slate-400 bg-slate-700 px-2 py-1 rounded">
            {filteredAndSortedPlayers.length} players
          </span>
        </div>

        <div className="flex flex-col sm:flex-row gap-4">
          {/* Search */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-slate-400" />
            <input
              type="text"
              placeholder="Search players or teams..."
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value);
                setCurrentPage(1);
              }}
              className="pl-10 pr-4 py-2 bg-slate-700 border border-slate-600 rounded-md text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent w-full sm:w-64"
            />
          </div>

          {/* Position Filter */}
          <select
            value={positionFilter}
            onChange={(e) => {
              setPositionFilter(e.target.value);
              setCurrentPage(1);
            }}
            className="px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="all">All Positions</option>
            <option value="Goalkeeper">Goalkeeper</option>
            <option value="Defender">Defender</option>
            <option value="Midfielder">Midfielder</option>
            <option value="Forward">Forward</option>
          </select>

          {/* Reset Filters */}
          {(searchTerm || positionFilter !== 'all') && (
            <button
              onClick={resetFilters}
              className="flex items-center space-x-2 px-3 py-2 bg-slate-600 hover:bg-slate-500 text-white rounded-md transition-colors"
            >
              <Filter className="h-4 w-4" />
              <span>Reset</span>
            </button>
          )}
        </div>
      </div>

      {/* Table */}
      <div className="bg-slate-800 rounded-lg border border-slate-700 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-slate-700">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-slate-300 uppercase tracking-wider">
                  Player
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-slate-300 uppercase tracking-wider">
                  Team
                </th>
                <th
                  className="px-6 py-3 text-left text-xs font-medium text-slate-300 uppercase tracking-wider cursor-pointer hover:text-white"
                  onClick={() => handleSort('position')}
                >
                  <div className="flex items-center space-x-1">
                    <span>Position</span>
                    {getSortIcon('position')}
                  </div>
                </th>
                <th
                  className="px-6 py-3 text-left text-xs font-medium text-slate-300 uppercase tracking-wider cursor-pointer hover:text-white"
                  onClick={() => handleSort('price')}
                >
                  <div className="flex items-center space-x-1">
                    <span>Price</span>
                    {getSortIcon('price')}
                  </div>
                </th>
                <th
                  className="px-6 py-3 text-left text-xs font-medium text-slate-300 uppercase tracking-wider cursor-pointer hover:text-white"
                  onClick={() => handleSort('predicted_xP')}
                >
                  <div className="flex items-center space-x-1">
                    <span>xP</span>
                    {getSortIcon('predicted_xP')}
                  </div>
                </th>
                <th
                  className="px-6 py-3 text-left text-xs font-medium text-slate-300 uppercase tracking-wider cursor-pointer hover:text-white"
                  onClick={() => handleSort('form')}
                >
                  <div className="flex items-center space-x-1">
                    <span>Form</span>
                    {getSortIcon('form')}
                  </div>
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-600">
              {paginatedPlayers.map((player) => (
                <tr key={player.id} className="hover:bg-slate-700 transition-colors">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <div className="flex-shrink-0 h-10 w-10">
                        <div className={`h-10 w-10 rounded-full ${getPositionColor(player.position)} flex items-center justify-center text-white font-bold text-xs`}>
                          {player.name.charAt(0)}
                        </div>
                      </div>
                      <div className="ml-4">
                        <div className="text-sm font-medium text-white">{player.name}</div>
                        <div className="text-sm text-slate-400">
                          {player.status === 'a' ? 'Available' : player.status === 'i' ? 'Injured' : 'Unavailable'}
                        </div>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-white">{player.team_name}</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`inline-flex px-2 py-1 text-xs font-medium rounded-full ${
                      player.position === 'Goalkeeper' ? 'bg-yellow-500/20 text-yellow-400' :
                      player.position === 'Defender' ? 'bg-blue-500/20 text-blue-400' :
                      player.position === 'Midfielder' ? 'bg-green-500/20 text-green-400' :
                      'bg-red-500/20 text-red-400'
                    }`}>
                      {player.position}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-white">
                    £{(player.price || 0).toFixed(1)}M
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`inline-flex px-2 py-1 text-xs font-medium rounded-full ${
                      (player.predicted_xP || 0) >= 5 ? 'bg-green-500/20 text-green-400' :
                      (player.predicted_xP || 0) >= 3 ? 'bg-yellow-500/20 text-yellow-400' :
                      'bg-red-500/20 text-red-400'
                    }`}>
                      {(player.predicted_xP || 0).toFixed(1)}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`inline-flex px-2 py-1 text-xs font-medium rounded-full ${
                      (player.form || 0) >= 5 ? 'bg-green-500/20 text-green-400' :
                      (player.form || 0) >= 3 ? 'bg-yellow-500/20 text-yellow-400' :
                      'bg-red-500/20 text-red-400'
                    }`}>
                      {(player.form || 0).toFixed(1)}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Empty State */}
        {paginatedPlayers.length === 0 && (
          <div className="text-center py-12">
            <Users className="h-12 w-12 text-slate-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-white mb-2">No Players Found</h3>
            <p className="text-slate-400">
              {searchTerm || positionFilter !== 'all'
                ? 'Try adjusting your search or filter criteria.'
                : 'No players data available.'
              }
            </p>
          </div>
        )}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <div className="text-sm text-slate-400">
            Showing {((currentPage - 1) * itemsPerPage) + 1} to {Math.min(currentPage * itemsPerPage, filteredAndSortedPlayers.length)} of {filteredAndSortedPlayers.length} players
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
              disabled={currentPage === 1}
              className="px-3 py-1 bg-slate-700 hover:bg-slate-600 disabled:bg-slate-800 disabled:opacity-50 text-white rounded-md transition-colors"
            >
              Previous
            </button>

            {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
              const pageNum = Math.max(1, Math.min(totalPages - 4, currentPage - 2)) + i;
              return (
                <button
                  key={pageNum}
                  onClick={() => setCurrentPage(pageNum)}
                  className={`px-3 py-1 rounded-md transition-colors ${
                    pageNum === currentPage
                      ? 'bg-blue-600 text-white'
                      : 'bg-slate-700 hover:bg-slate-600 text-white'
                  }`}
                >
                  {pageNum}
                </button>
              );
            })}

            <button
              onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
              disabled={currentPage === totalPages}
              className="px-3 py-1 bg-slate-700 hover:bg-slate-600 disabled:bg-slate-800 disabled:opacity-50 text-white rounded-md transition-colors"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default PlayersTable;
