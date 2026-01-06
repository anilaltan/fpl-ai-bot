import React from 'react';

const SoccerPitch = ({ players }) => {
  // Separate starting XI and bench players
  const startingPlayers = players.filter(player => player.is_starting === true);
  const benchPlayers = players.filter(player => player.is_starting !== true);

  // Group starting players by position for pitch display
  const groupedPlayers = {
    GKP: startingPlayers.filter(player => player.position === 'GKP'),
    DEF: startingPlayers.filter(player => player.position === 'DEF'),
    MID: startingPlayers.filter(player => player.position === 'MID'),
    FWD: startingPlayers.filter(player => player.position === 'FWD'),
  };

  // Player card component
  const PlayerCard = ({ player, compact = false }) => {
    const xpValue = player.predicted_xP || player.xp || 0;
    const isHighXp = xpValue >= 6.0;
    const isCaptain = player.is_captain;
    const isViceCaptain = player.is_vice_captain;

    // For captain, show 2x points with special styling
    const displayXp = isCaptain ? xpValue * 2 : xpValue;
    const xpColor = isCaptain ? 'text-yellow-500 font-extrabold' :
                  isHighXp ? 'text-yellow-600 font-bold' : 'text-blue-600';

    return (
      <div className={`relative bg-white rounded-lg p-2 shadow-lg border-2 transition-all duration-200 hover:scale-105 ${
        isCaptain ? 'border-yellow-500 shadow-yellow-500/50 shadow-lg bg-gradient-to-br from-yellow-50 to-yellow-100' :
        isHighXp ? 'border-yellow-400 shadow-yellow-400/50 shadow-lg' : 'border-gray-300'
      } ${compact ? 'w-16 h-16' : 'w-20 h-20'}`}>
        {/* Captain/Vice Captain Badge */}
        {(isCaptain || isViceCaptain) && (
          <div className={`absolute -top-2 -right-2 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold shadow-lg ${
            isCaptain ? 'bg-yellow-500 text-black border-2 border-yellow-300' :
            'bg-blue-500 text-white border-2 border-blue-300'
          }`}>
            {isCaptain ? 'C' : 'V'}
          </div>
        )}

        <div className="flex flex-col items-center justify-center h-full text-xs">
          <div className="font-bold text-gray-900 text-center leading-tight">
            {player.name.split(' ').pop()} {/* Last name only */}
          </div>
          <div className="text-gray-600 text-center mt-0.5">
            {player.team_name?.substring(0, 3).toUpperCase() || 'UNK'}
          </div>
          <div className="text-green-600 font-semibold mt-0.5">
            £{player.price?.toFixed(1) || '0.0'}M
          </div>
          <div className={`mt-0.5 ${xpColor}`}>
            {displayXp.toFixed(1)}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="w-full max-w-4xl mx-auto mb-8">
      {/* Starting XI - Football Pitch */}
      <div className="relative bg-gradient-to-b from-green-800 to-green-900 rounded-t-2xl p-6 border-4 border-white border-b-0 shadow-2xl overflow-hidden h-96">
        {/* Pitch Lines Effect */}
        <div className="absolute inset-0 opacity-20">
          {/* Center Circle */}
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-32 h-32 border-2 border-white rounded-full"></div>
          {/* Center Line */}
          <div className="absolute top-1/2 left-0 right-0 h-0.5 bg-white"></div>
          {/* Penalty Areas */}
          <div className="absolute top-8 left-8 right-8 h-20 border-2 border-white rounded-t-xl"></div>
          <div className="absolute bottom-8 left-8 right-8 h-20 border-2 border-white rounded-b-xl"></div>
        </div>

        {/* Starting XI Players Layout */}
        <div className="absolute inset-0 flex flex-col justify-around py-6 z-10">
          {/* Forwards Row (Top of pitch) */}
          {groupedPlayers.FWD.length > 0 && (
            <div className="flex justify-center">
              <div className="flex space-x-3">
                {groupedPlayers.FWD.map((player, index) => (
                  <PlayerCard key={`fwd-${player.id}-${index}`} player={player} compact={groupedPlayers.FWD.length > 3} />
                ))}
              </div>
            </div>
          )}

          {/* Midfielders Row */}
          {groupedPlayers.MID.length > 0 && (
            <div className="flex justify-center">
              <div className="flex space-x-3 flex-wrap justify-center max-w-md">
                {groupedPlayers.MID.map((player, index) => (
                  <PlayerCard key={`mid-${player.id}-${index}`} player={player} compact={groupedPlayers.MID.length > 4} />
                ))}
              </div>
            </div>
          )}

          {/* Defenders Row */}
          {groupedPlayers.DEF.length > 0 && (
            <div className="flex justify-center">
              <div className="flex space-x-3 flex-wrap justify-center max-w-lg">
                {groupedPlayers.DEF.map((player, index) => (
                  <PlayerCard key={`def-${player.id}-${index}`} player={player} compact={groupedPlayers.DEF.length > 4} />
                ))}
              </div>
            </div>
          )}

          {/* Goalkeeper Row (Bottom of pitch) */}
          {groupedPlayers.GKP.length > 0 && (
            <div className="flex justify-center">
              <div className="flex space-x-3">
                {groupedPlayers.GKP.map((player, index) => (
                  <PlayerCard key={`gkp-${player.id}-${index}`} player={player} />
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Pitch Title */}
        <div className="absolute top-4 left-1/2 transform -translate-x-1/2">
          <h3 className="text-white font-bold text-lg bg-black/60 px-4 py-2 rounded-lg border border-white/20">
            Starting XI
          </h3>
        </div>

        {/* Formation Label */}
        <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2">
          <div className="text-white/80 text-sm bg-black/40 px-3 py-1 rounded-md">
            {groupedPlayers.GKP.length}-{groupedPlayers.DEF.length}-{groupedPlayers.MID.length}-{groupedPlayers.FWD.length} Formation
          </div>
        </div>
      </div>

      {/* Bench/Substitutes Area */}
      {benchPlayers.length > 0 && (
        <div className="relative bg-gradient-to-b from-gray-800 to-gray-900 rounded-b-2xl p-6 border-4 border-white border-t-0 shadow-2xl overflow-hidden">
          {/* Bench Title */}
          <div className="absolute top-4 left-1/2 transform -translate-x-1/2">
            <h3 className="text-white font-bold text-lg bg-black/60 px-4 py-2 rounded-lg border border-white/20">
              Bench ({benchPlayers.length} players)
            </h3>
          </div>

          {/* Bench Players Layout */}
          <div className="flex justify-center items-center h-full pt-12">
            <div className="flex space-x-4 flex-wrap justify-center max-w-4xl">
              {benchPlayers.map((player, index) => (
                <PlayerCard key={`bench-${player.id}-${index}`} player={player} compact={benchPlayers.length > 4} />
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="flex justify-center mt-4 space-x-8 text-sm text-gray-400">
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 bg-yellow-400 border border-yellow-300 rounded"></div>
          <span>High xP (≥6.0)</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 bg-yellow-500 border-2 border-yellow-300 rounded-full flex items-center justify-center">
            <span className="text-xs font-bold text-black">C</span>
          </div>
          <span>Captain (2x points)</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 bg-blue-500 border-2 border-blue-300 rounded-full flex items-center justify-center">
            <span className="text-xs font-bold text-white">V</span>
          </div>
          <span>Vice Captain</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 bg-gray-300 border border-gray-400 rounded"></div>
          <span>Normal xP</span>
        </div>
      </div>
    </div>
  );
};

export default SoccerPitch;
