import React from 'react';
import PlayerCard from './PlayerCard';

const PitchView = ({ team }) => {
  // Group players by position
  const groupedPlayers = {
    GK: team.filter(player => player.position === 'GK'),
    DEF: team.filter(player => player.position === 'DEF'),
    MID: team.filter(player => player.position === 'MID'),
    FWD: team.filter(player => player.position === 'FWD'),
  };

  // Calculate positions for each row (centered layout)
  const getRowLayout = (players, maxPerRow = 5) => {
    if (players.length === 0) return [];
    const rows = [];
    for (let i = 0; i < players.length; i += maxPerRow) {
      rows.push(players.slice(i, i + maxPerRow));
    }
    return rows;
  };

  const gkRows = getRowLayout(groupedPlayers.GK, 1);
  const defRows = getRowLayout(groupedPlayers.DEF, 5);
  const midRows = getRowLayout(groupedPlayers.MID, 5);
  const fwdRows = getRowLayout(groupedPlayers.FWD, 3);

  return (
    <div className="w-full max-w-4xl mx-auto">
      {/* Football Pitch Container */}
      <div className="relative bg-gradient-to-b from-slate-900 to-emerald-950/30 rounded-2xl p-6 border border-slate-700 overflow-hidden">
        {/* Pitch Lines Effect */}
        <div className="absolute inset-0 opacity-10">
          {/* Center Circle */}
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-32 h-32 border-2 border-emerald-400 rounded-full"></div>
          {/* Center Line */}
          <div className="absolute top-1/2 left-0 right-0 h-px bg-emerald-400"></div>
          {/* Penalty Areas */}
          <div className="absolute top-8 left-8 right-8 h-20 border-2 border-emerald-400 rounded-t-xl"></div>
          <div className="absolute bottom-8 left-8 right-8 h-20 border-2 border-emerald-400 rounded-b-xl"></div>
        </div>

        {/* Players Layout */}
        <div className="relative z-10 space-y-8">
          {/* Goalkeeper Row */}
          {gkRows.map((row, rowIndex) => (
            <div key={`gk-${rowIndex}`} className="flex justify-center">
              <div className="flex space-x-4">
                {row.map((player, index) => (
                  <PlayerCard key={`gk-${player.id}-${index}`} player={player} compact={true} />
                ))}
              </div>
            </div>
          ))}

          {/* Defender Rows */}
          {defRows.map((row, rowIndex) => (
            <div key={`def-${rowIndex}`} className="flex justify-center">
              <div className="flex space-x-4">
                {row.map((player, index) => (
                  <PlayerCard key={`def-${player.id}-${index}`} player={player} compact={true} />
                ))}
              </div>
            </div>
          ))}

          {/* Midfielder Rows */}
          {midRows.map((row, rowIndex) => (
            <div key={`mid-${rowIndex}`} className="flex justify-center">
              <div className="flex space-x-4">
                {row.map((player, index) => (
                  <PlayerCard key={`mid-${player.id}-${index}`} player={player} compact={true} />
                ))}
              </div>
            </div>
          ))}

          {/* Forward Rows */}
          {fwdRows.map((row, rowIndex) => (
            <div key={`fwd-${rowIndex}`} className="flex justify-center">
              <div className="flex space-x-4">
                {row.map((player, index) => (
                  <PlayerCard key={`fwd-${player.id}-${index}`} player={player} compact={true} />
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Pitch Title */}
        <div className="absolute top-4 left-1/2 transform -translate-x-1/2">
          <h3 className="text-white font-bold text-lg bg-slate-900/80 px-4 py-2 rounded-lg border border-slate-700">
            Dream Team
          </h3>
        </div>
      </div>

      {/* Team Stats */}
      {team.length > 0 && (
        <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-slate-800 rounded-lg p-4 border border-slate-700 text-center">
            <div className="text-2xl font-bold text-white">{team.length}</div>
            <div className="text-slate-400">Players</div>
          </div>
          <div className="bg-slate-800 rounded-lg p-4 border border-slate-700 text-center">
            <div className="text-2xl font-bold text-emerald-400">
              {team.reduce((sum, player) => sum + (player.risk_adjusted_xP || 0), 0).toFixed(1)}
            </div>
            <div className="text-slate-400">Total xP</div>
          </div>
          <div className="bg-slate-800 rounded-lg p-4 border border-slate-700 text-center">
            <div className="text-2xl font-bold text-blue-400">
              £{team.reduce((sum, player) => sum + (player.price || 0), 0).toFixed(1)}M
            </div>
            <div className="text-slate-400">Total Cost</div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PitchView;
