"""Window/graphing variable side effects and the Zoom memory."""
import pytest

from environment import Environment, TableVars
from catalog import get_token, CommandToken
from errors import DomainError
from test_tibasic import run

# Token codes for the window/table variables (from catalog).
DELTA_X, DELTA_Y = 0x6326, 0x6327
XRES             = 0x6336
NMIN, NMAX       = 0x631F, 0x631D
XFACT, YFACT     = 0x6328, 0x6329
TBL_START        = 0x631A
ZXMIN, ZXMAX     = 0x6312, 0x6313
ZNMIN, ZXRES     = 0x6320, 0x6337


def store(env, code, value):
	"""Store through the exact path the parser uses for `value→var`."""
	get_token(code).store(env, value)


def resolve(env, code):
	return get_token(code).resolve(env)


class TestDeltaDerived:
	def test_default_is_bounds_over_94(self):
		env = Environment()
		assert resolve(env, DELTA_X) == pytest.approx((10 - -10) / 94)

	def test_delta_y_over_62(self):
		env = Environment()
		assert resolve(env, DELTA_Y) == pytest.approx((10 - -10) / 62)

	def test_tracks_live_bounds(self):
		env = Environment()
		env.window.xmin = 0.0
		env.window.xmax = 94.0
		assert resolve(env, DELTA_X) == pytest.approx(1.0)

	def test_store_delta_x_moves_xmax_only(self):
		env = Environment()
		store(env, DELTA_X, 1.0)                      # xmax := xmin + 94·ΔX
		assert env.window.xmax == pytest.approx(-10 + 94)
		assert env.window.xmin == -10                  # lower bound untouched

	def test_store_delta_y_moves_ymax_only(self):
		env = Environment()
		store(env, DELTA_Y, 2.0)
		assert env.window.ymax == pytest.approx(-10 + 62 * 2)
		assert env.window.ymin == -10

	def test_round_trips(self):
		env = Environment()
		store(env, DELTA_X, 0.5)
		assert resolve(env, DELTA_X) == pytest.approx(0.5)


class TestXres:
	def test_accepts_integer_in_range(self):
		env = Environment()
		store(env, XRES, 4.0)
		assert resolve(env, XRES) == 4.0

	@pytest.mark.parametrize("bad", [0.0, 9.0, -1.0, 1.5])
	def test_rejects_out_of_range_or_non_integer(self, bad):
		env = Environment()
		with pytest.raises(DomainError):
			store(env, XRES, bad)


class TestNMinNMax:
	def test_accepts_integers(self):
		env = Environment()
		store(env, NMIN, 3.0)
		store(env, NMAX, 20.0)
		assert (resolve(env, NMIN), resolve(env, NMAX)) == (3.0, 20.0)

	def test_rejects_non_integer(self):
		env = Environment()
		with pytest.raises(DomainError):
			store(env, NMAX, 2.5)

	def test_stays_float(self):
		env = Environment()
		store(env, NMIN, 5.0)
		assert type(resolve(env, NMIN)) is float


class TestZoomFactors:
	@pytest.mark.parametrize("code", [XFACT, YFACT])
	def test_accepts_at_least_one(self, code):
		env = Environment()
		store(env, code, 1.0)        # boundary
		store(env, code, 7.5)
		assert resolve(env, code) == 7.5

	@pytest.mark.parametrize("code", [XFACT, YFACT])
	def test_rejects_below_one(self, code):
		env = Environment()
		with pytest.raises(DomainError):
			store(env, code, 0.5)


class TestZoomMemory:
	def test_sto_then_rcl_restores_window(self):
		env = Environment()
		env.window.xmax = 50.0
		run('ZoomSto', env)
		env.window.xmax = 99.0
		run('ZoomRcl', env)
		assert env.window.xmax == 50.0

	def test_rcl_restores_derived_delta(self):
		env = Environment()
		env.window.xmin = 0.0
		env.window.xmax = 94.0   # ΔX = 1
		run('ZoomSto', env)
		env.window.xmax = 10.0
		run('ZoomRcl', env)
		assert resolve(env, DELTA_X) == pytest.approx(1.0)

	def test_store_snapshot_is_independent(self):
		env = Environment()
		env.window.xmax = 50.0
		run('ZoomSto', env)
		env.window.xmax = 99.0
		assert env.window.zxmax == 50.0   # z-var unchanged after live window mutated


class TestZWindowVars:
	"""The Z-window system variables are real, addressable variables on env.window."""

	def test_zoomsto_populates_z_vars(self):
		env = Environment()
		env.window.xmin, env.window.xmax = -3.0, 7.0
		run('ZoomSto', env)
		assert resolve(env, ZXMIN) == -3.0
		assert resolve(env, ZXMAX) == 7.0

	def test_store_to_z_var_hits_window(self):
		env = Environment()
		store(env, ZXMIN, 12.0)
		assert env.window.zxmin == 12.0
		assert env.window.xmin == -10.0          # live window untouched

	def test_zoomrcl_restores_from_z_vars(self):
		# Writing the Z-variables directly, then ZoomRcl, drives the live window.
		env = Environment()
		store(env, ZXMIN, 5.0)
		store(env, ZXMAX, 25.0)
		run('ZoomRcl', env)
		assert (env.window.xmin, env.window.xmax) == (5.0, 25.0)

	def test_z_vars_keep_their_counterpart_validation(self):
		env = Environment()
		with pytest.raises(DomainError):
			store(env, ZXRES, 9.0)               # ZXres is still an integer 1-8
		with pytest.raises(DomainError):
			store(env, ZNMIN, 2.5)               # ZnMin is still whole-number


class TestTableVarsSeparated:
	def test_table_token_routes_to_table_not_window(self):
		env = Environment()
		store(env, TBL_START, 5.0)
		assert env.table.tbl_start == 5.0
		assert not hasattr(env.window, 'tbl_start')

	def test_tbl_input_is_a_list_slot(self):
		# TblInput is a plain TiList | None slot, defaulting to None (no token wires it yet).
		assert TableVars().tbl_input is None


class TestWiring:
	def test_zoom_commands_are_wired(self):
		assert isinstance(get_token(0x92), CommandToken)   # ZoomSto
		assert isinstance(get_token(0x90), CommandToken)   # ZoomRcl
