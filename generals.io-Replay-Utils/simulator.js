'use strict';

const MAP_UNKNOWN = -1;
const MAP_UNKNOWN_OBSTACLE = -2;
const MAP_MOUNTAIN = -3;
const MAP_EMPTY = -4;

var fs = require('fs');
var Game = require('./Game');

var isValue = false;
if (process.argv.length >= 3)
{
	var type = process.argv[2];
	if (type != "policy" && type != "value")
	{
		throw new Error();
	}
	isValue = (type === "value")
}

var replay_paths = require('fs').readFileSync("/dev/stdin", 'utf-8')
    .split('\n')
    .map(function(s) { return s.trim(); })
    .filter(Boolean);

function path_to_replay(path) {
	return JSON.parse(fs.readFileSync(path, 'utf8'));
}

function simulate(replay, gameCallback, moveCallback) {
	var game = Game.createFromReplay(replay);

	var currentMoveIndex = 0;
	var currentAFKIndex = 0;

	function nextTurn() {
		// Put moves in the move queue.
		while (replay.moves.length > currentMoveIndex && replay.moves[currentMoveIndex].turn <= game.turn) {
			var move = replay.moves[currentMoveIndex++];
			moveCallback(move);
			game.handleAttack(move.index, move.start, move.end, move.is50);
		}

		// Check for AFKs.
		while (replay.afks.length > currentAFKIndex && replay.afks[currentAFKIndex].turn <= game.turn) {
			var afk = replay.afks[currentAFKIndex++];
			var index = afk.index;

			// If already dead, mark as dead general and neutralize if needed.
			if (game.deaths.indexOf(game.sockets[index]) >= 0) {
				game.tryNeutralizePlayer(index);
			}
			// Mark as AFK if not already dead.
			else {
				game.deaths.push(game.sockets[index]);
				game.alivePlayers--;
			}
		}

		game.update();
	}

	// Simulate the game!
	while (!game.isOver() && game.turn < 2000) {
		nextTurn();
		if (!game.isOver())
		{
			gameCallback(game);
		}
		// game.turn: The current turn.
		// game.sockets: The array of players. Player game.sockets[i] has playerIndex i.
		// game.map: A Map object representing the current game state. See Map.js.
		// game.scores: An ordered (decreasing) array of scores. Each score object can be tied to a player by its .i field.
		// game.alivePlayers: The number of players left alive.
		// game.deaths: Dead players in chronological order: game.deaths[0] is the first player to die.
	}
	return game;
}

function winner(game) {
	var alive = game.scores.filter(function(x) { return !x.dead; });
	if (alive.length != 1)
	{
		throw new Error("Expected 1 living player, found " + alive.length);
	}
	return alive[0].i;
}

function print_observed_map(observe) {
	var m = observe.mapOwner;
	for (var r = 0; r < observe.mapHeight; r++)
	{
		var s = "";
		for (var c = 0; c < observe.mapWidth; c++)
		{
			var o = m[r*observe.mapWidth+c];
			if (o == MAP_UNKNOWN)
				s += '?';
			else if (o == MAP_UNKNOWN_OBSTACLE)
				s += '>';
			else if (o == MAP_MOUNTAIN)
				s += 'X';
			else if (o == MAP_EMPTY)
				s += '.';
			else
				s += o;
		}
		console.log(s);
	}
	console.log("");
}

function game_to_observe(game, perspective) {
	if (isValue)
	{
		game.map.indexVisibleTo = function(a, b) { return true; }
	}
	if (perspective !== 0 && perspective !== 1)
	{
		throw new Error("Invalid perspective value: " + perspective);
	}
	if (game.scores.length != 2)
	{
		throw new Error("Expected 2 players, found " + game.scores.length);
	}
	var opponent = 1 - perspective;
	var size = game.map.width * game.map.height;
	var result = {
		type: 'observe',
		turn: game.turn,
		mapWidth: game.map.width,
		mapHeight: game.map.height,
		mapOwner: new Array(size),
		mapForces: new Array(size),
		cities: [],
		generals: [],
		forces: new Array(2),
		land: new Array(2)
	};

	// result.mapOwner
	for (var i = 0; i < size; i++)
	{
		result.mapForces[i] = 0;
		if (game.map.indexVisibleTo(i, perspective))
		{
			var owner = game.map.tileAt(i);
			if (owner == -1)
			{
				result.mapOwner[i] = MAP_EMPTY;
			}
			else if (owner == -2)
			{
				result.mapOwner[i] = MAP_MOUNTAIN;
			}
			else
			{
				result.mapOwner[i] = Math.abs(owner - perspective);
				result.mapForces[i] = game.map.armyAt(i);
			}
		}
		else
		{
			if (game.map.tileAt(i) == -2 || game.isCity(i))
			{
				result.mapOwner[i] = MAP_UNKNOWN_OBSTACLE;
			}
			else
			{
				result.mapOwner[i] = MAP_UNKNOWN;
			}
		}
	}

	// result.generals
	if (!game.map.indexVisibleTo(game.generals[perspective], perspective))
	{
		throw new Error("Can't see our own general");
	}
	result.generals.push(game.generals[perspective]);
	if (game.map.indexVisibleTo(game.generals[opponent], perspective))
	{
		result.generals.push(game.generals[opponent]);
	}

	// result.cities
	game.cities.forEach(function(city) {
		if (game.map.indexVisibleTo(city, perspective))
		{
			result.cities.push(city);
		}
	});
	result.generals.forEach(function(general) {
		result.cities.push(general);
	});

	// result.forces, result.land
	game.scores.forEach(function(score) {
		result.forces[Math.abs(score.i - perspective)] = score.total;
		result.land[Math.abs(score.i - perspective)] = score.tiles;
	});

	if (isValue)
	{
		result.mapOwner.forEach(function(x) {
			if (x == MAP_UNKNOWN || x == MAP_UNKNOWN_OBSTACLE)
			{
				throw new Error("isValue=true but not all tiles are visible")
			}
		});
	}

	// print_observed_map(result);

	return result;
}

function emptyMove() {
	return {
		type: 'move',
		emptyMove: true
	}
}

function moveBetween(src, dst) {
	return {
		type: 'move',
		emptyMove: false,
		from: src,
		to: dst
	};
}

function replay_to_eseq(replay) {
	var finalState = simulate(
		replay,
		function(x) {},
		function(x) {});
	var victor = winner(finalState);
	var perspective;
	if (isValue)
		perspective = 0;
	else
		perspective = victor;
	var metadata = {
		type: 'metadata',
		replay_id: replay['id'],
		won: (perspective === victor)
	};
	var eseq = [metadata];
	simulate(
		replay,
		function(game) {
			if (eseq.length > 0 && eseq[eseq.length - 1].type == 'observe')
			{
				eseq.push(emptyMove());
			}
			eseq.push(game_to_observe(game, perspective));
		},
		function(move) {
			if (move.index == perspective && eseq.length > 0 && eseq[eseq.length - 1].type == 'observe')
			{
				eseq.push(moveBetween(move.start, move.end));
			}
		});
	return eseq;
}

var prevPct = 0;
var cnt = 0;
var tot = replay_paths.length;
replay_paths.forEach(function(path) {
	try {
		var replay = path_to_replay(path);
		var eseq = replay_to_eseq(replay);
		console.log(JSON.stringify(eseq));
	} catch (e) {
		console.error("simulator.js: Encountered error `" + e + "` while processing " + path)
	}
	++cnt;
	var newPct = cnt / tot;
	if (Math.floor(newPct * 50) != Math.floor(prevPct * 50))
	{
		console.error("simulator.js: " + cnt + "/" + tot + " replays processed");
	}
	prevPct = newPct;
});
