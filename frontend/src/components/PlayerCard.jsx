import React from 'react';

const PlayerCard = ({ player, compact = false }) => {
  // Calculate progress bar percentages (assuming scores are between 0-10 or similar)
  const maxScore = 10; // Adjust based on your scoring system
  const techProgress = player.technical_score ? (player.technical_score / maxScore) * 100 : 0;
  const marketProgress = player.market_score ? (player.market_score / maxScore) * 100 : 0;
  const tactProgress = player.tactical_score ? (player.tactical_score / maxScore) * 100 : 0;

  return (
    <div className={`
      bg-slate-800 border border-slate-700 rounded-xl p-4
      transition-all duration-200 hover:-translate-y-1 hover:shadow-lg hover:shadow-slate-900/50
      ${compact ? 'scale-90' : ''}
    `}>
      {/* Player Info */}
      <div className="text-center mb-3">
        <h3 className={`font-bold text-white ${compact ? 'text-sm' : 'text-lg'} mb-1`}>
          {player.web_name}
        </h3>
        <p className="text-xs text-slate-400 mb-1">{player.team_name}</p>
        <span className={`
          inline-block px-2 py-1 rounded-full text-xs font-medium
          ${player.position === 'GK' ? 'bg-yellow-500/20 text-yellow-400' :
            player.position === 'DEF' ? 'bg-blue-500/20 text-blue-400' :
            player.position === 'MID' ? 'bg-green-500/20 text-green-400' :
            'bg-red-500/20 text-red-400'}
        `}>
          {player.position}
        </span>
      </div>

      {/* Total xP Value */}
      <div className="text-center mb-4">
        <div className={`text-emerald-400 font-bold ${compact ? 'text-xl' : 'text-2xl'}`}>
          {player.risk_adjusted_xP ? player.risk_adjusted_xP.toFixed(1) : 'N/A'}
        </div>
        <div className="text-xs text-slate-400">Total xP</div>
      </div>

      {/* Ensemble Score Bars */}
      {!compact && (
        <div className="space-y-2">
          {/* Technical Score */}
          <div>
            <div className="flex justify-between items-center mb-1">
              <span className="text-xs text-slate-400">Tech</span>
              <span className="text-xs text-blue-400">
                {player.technical_score ? player.technical_score.toFixed(1) : 'N/A'}
              </span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-1.5">
              <div
                className="bg-blue-500 h-1.5 rounded-full transition-all duration-500"
                style={{ width: `${Math.min(techProgress, 100)}%` }}
              ></div>
            </div>
          </div>

          {/* Market Score */}
          <div>
            <div className="flex justify-between items-center mb-1">
              <span className="text-xs text-slate-400">Market</span>
              <span className="text-xs text-emerald-400">
                {player.market_score ? player.market_score.toFixed(1) : 'N/A'}
              </span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-1.5">
              <div
                className="bg-emerald-500 h-1.5 rounded-full transition-all duration-500"
                style={{ width: `${Math.min(marketProgress, 100)}%` }}
              ></div>
            </div>
          </div>

          {/* Tactical Score */}
          <div>
            <div className="flex justify-between items-center mb-1">
              <span className="text-xs text-slate-400">Tact</span>
              <span className="text-xs text-rose-400">
                {player.tactical_score ? player.tactical_score.toFixed(1) : 'N/A'}
              </span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-1.5">
              <div
                className="bg-rose-500 h-1.5 rounded-full transition-all duration-500"
                style={{ width: `${Math.min(tactProgress, 100)}%` }}
              ></div>
            </div>
          </div>
        </div>
      )}

      {/* Captain Indicator */}
      {player.is_captain && (
        <div className="mt-3 text-center">
          <span className="inline-block px-2 py-1 bg-yellow-500/20 text-yellow-400 text-xs font-bold rounded-full">
            CAPTAIN
          </span>
        </div>
      )}
    </div>
  );
};

export default PlayerCard;
