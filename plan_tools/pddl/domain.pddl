(define (domain ltamp)
    (:requirements :strips :equality)
    (:constants @left @right @left-conf @right-conf @coffee @sugar)
    (:predicates
        ; Static predicates (predicates that do not change over time)
        (IsMove ?arm ?conf1 ?conf2 ?control)
        (IsPick ?arm ?obj ?pose ?grasp ?conf1 ?conf2 ?control)
        (IsPlace ?arm ?obj ?pose ?grasp ?conf1 ?conf2 ?control)
        (IsPour ?arm ?bowl ?pose ?cup ?grasp ?conf1 ?conf2 ?control)
        (IsPress ?arm ?obj ?conf1 ?conf2 ?control)
        (IsPush ?arm ?obj ?pose1 ?pose2 ?conf1 ?conf2 ?control)
        (IsStir ?arm ?bowl ?pose ?stirrer ?grasp ?conf1 ?conf2 ?control)
        (IsScoop ?arm ?bowl ?pose ?spoon ?grasp ?conf1 ?conf2 ?control)
        ; (WillSpill ?arm ?conf ?obj ?grasp)

        (CanHold ?bowl)
        (CanPush ?obj ?region)
        (Pourable ?cup)
        (Pressable ?button)
        (Scoopable ?bowl)
        (CanScoop ?spoon)
        (Stirrable ?stirrer)
        (Material ?material)

        ; Fluent predicates (predicates that change over time, which describes the state of the system)
        (IsArm ?arm)
        (IsPose ?obj ?pose)
        (IsGrasp ?obj ?grasp)
        (IsConf ?arm ?conf)
        (IsControl ?arm ?control)
        (IsSupported ?obj ?pose ?surface ?pose2 ?link)
        (Contained ?obj ?pose ?region)
        (Movable ?obj)
        (Graspable ?obj)
        (IsType ?obj ?ty)
        (Stackable ?obj ?surface ?link)
        (Stove ?surface ?link)
        (Cooked ?obj)
        (Mixed ?bowl)
        (IsButton ?button ?device)

        (CanMove ?arm)
        (AtConf ?arm ?conf)
        (AtPose ?obj ?pose)
        (AtGrasp ?arm ?obj ?grasp)
        (Empty ?arm)
        (Grasped ?arm ?obj)
        (On ?obj ?surface ?link)
        (InRegion ?obj ?region)
        (Contains ?obj ?material)
        (Supporting ?surface)

        ; Derived predicates (predicates derived from other predicates, defined with streams)
        (UnsafeControl ?arm ?control)
        (UnsafeConf ?arm ?conf)
        (UnsafePose ?obj ?pose)
        (Holding ?obj)
        (HoldingType ?ty)

        ; External predicates (evaluated by external boolean functions)
        (ControlPoseCollision ?arm ?control ?obj ?pose)
        (ConfConfCollision ?arm ?control ?obj ?pose)
        (ControlConfCollision ?arm ?control ?arm2 ?conf2)
        (PosePoseCollision ?obj ?pose ?obj2 ?pose2)
    )
    ; TODO: unify AtPose and AtConf
    ; TODO: make water/mixture an object that can be heated/cooled

    (:action move-arm
        :parameters (?arm ?conf1 ?conf2 ?control)
        :precondition (and (IsMove ?arm ?conf1 ?conf2 ?control)
                           (or (= ?arm @left) (AtConf @left @left-conf))
                           (or (= ?arm @right) (AtConf @right @right-conf))
                           (AtConf ?arm ?conf1) ; (CanMove ?arm)
                           (not (UnsafeConf ?arm ?conf2)))
        :effect (and (AtConf ?arm ?conf2)
                     ; (forall (?arm2) (when (not (= ?arm ?arm2) (CanMove ?arm2))))
                     (not (AtConf ?arm ?conf1)) (not (CanMove ?arm)))
    )

    (:action pick
        :parameters (?arm ?obj ?pose ?grasp ?conf1 ?conf2 ?control)
        :precondition (and (IsPick ?arm ?obj ?pose ?grasp ?conf1 ?conf2 ?control)
                           (AtPose ?obj ?pose) (Empty ?arm) (AtConf ?arm ?conf1)
                           (not (UnsafeControl ?arm ?control)) (not (Supporting ?obj)))
        :effect (and (AtConf ?arm ?conf2) (AtGrasp ?arm ?obj ?grasp) (CanMove ?arm)
                     (not (AtPose ?obj ?pose)) (not (Empty ?arm)) (not (AtConf ?arm ?conf1))) ; (not (On ?obj ?surface ?link)))
    )

    (:action place
        :parameters (?arm ?obj ?pose ?grasp ?surface ?pose2 ?link ?conf1 ?conf2 ?control)
        :precondition (and (IsPlace ?arm ?obj ?pose ?grasp ?conf1 ?conf2 ?control) (IsSupported ?obj ?pose ?surface ?pose2 ?link)
                           (AtGrasp ?arm ?obj ?grasp) (AtPose ?surface ?pose2) (AtConf ?arm ?conf1)
                           (not (UnsafeControl ?arm ?control)))
        :effect (and (AtPose ?obj ?pose) (Empty ?arm) (AtConf ?arm ?conf2) (CanMove ?arm) ; (On ?obj ?surface ?link)
                     (not (AtGrasp ?arm ?obj ?grasp)) (not (AtConf ?arm ?conf1)))
    )

    (:action push
        :parameters (?arm ?obj ?pose1 ?pose2 ?conf1 ?conf2 ?control)
        :precondition (and (IsPush ?arm ?obj ?pose1 ?pose2 ?conf1 ?conf2 ?control)
                           (AtPose ?obj ?pose1) (Empty ?arm) (AtConf ?arm ?conf1)
                           (not (UnsafeControl ?arm ?control)) (not (Supporting ?obj)))
        :effect (and (AtPose ?obj ?pose2) (AtConf ?arm ?conf2) (CanMove ?arm)
                     (not (AtPose ?obj ?pose1)) (not (AtConf ?arm ?conf1)))
    )

    ;(:action pour
    ;    :parameters (?arm ?bowl ?pose ?cup ?grasp ?conf1 ?conf2 ?control)
    ;    :precondition (and (IsPour ?arm ?bowl ?pose ?cup ?grasp ?conf1 ?conf2 ?control)
    ;                       (AtPose ?bowl ?pose) (AtGrasp ?arm ?cup ?grasp) (AtConf ?arm ?conf1)
    ;                       (not (= ?bowl ?cup)) (not (UnsafeControl ?arm ?control)))
    ;    :effect (and (AtConf ?arm ?conf2) (CanMove ?arm) (not (AtConf ?arm ?conf1))
    ;                 ; TODO: FD bug that requires these to be different parameter names
    ;                 (forall (?material1) (when (and (Material ?material1) (Contains ?cup ?material1))
    ;                                            (Contains ?bowl ?material1)))
    ;                 (forall (?material2) (when (and (Material ?material2) (Contains ?cup ?material2))
    ;                                            (not (Contains ?cup ?material2)))))
    ;)

    (:action pour
        :parameters (?arm ?bowl ?pose ?cup ?grasp ?material ?conf1 ?conf2 ?control)
        :precondition (and (IsPour ?arm ?bowl ?pose ?cup ?grasp ?conf1 ?conf2 ?control)
                           (Material ?material) (Contains ?cup ?material)
                           (AtPose ?bowl ?pose) (AtGrasp ?arm ?cup ?grasp) (AtConf ?arm ?conf1)
                           (not (= ?bowl ?cup)) (not (UnsafeControl ?arm ?control)))
        :effect (and (AtConf ?arm ?conf2) (CanMove ?arm) (Contains ?bowl ?material)
                     (not (AtConf ?arm ?conf1)) (not (Contains ?cup ?material)))
    )

    ; TODO: example where you need to make two cups of coffee
    (:action scoop
        :parameters (?arm ?bowl ?pose ?spoon ?grasp ?material ?conf1 ?conf2 ?control)
        :precondition (and (IsScoop ?arm ?bowl ?pose ?spoon ?grasp ?conf1 ?conf2 ?control)
                           (Material ?material) (Contains ?bowl ?material)
                           (AtPose ?bowl ?pose) (AtGrasp ?arm ?spoon ?grasp) (AtConf ?arm ?conf1)
                           (not (UnsafeControl ?arm ?control)))
        :effect (and (AtConf ?arm ?conf2) (CanMove ?arm) (Contains ?spoon ?material)
                     (not (AtConf ?arm ?conf1))) ; Intentionally not deleting (Contains ?bowl ?material)
        ; TODO: ensure that not already occupied
    )

    ; TODO: ensure the spoon is empty
    (:action stir
        :parameters (?arm ?bowl ?pose ?stirrer ?grasp ?conf1 ?conf2 ?control)
        :precondition (and (IsStir ?arm ?bowl ?pose ?stirrer ?grasp ?conf1 ?conf2 ?control)
                           (AtPose ?bowl ?pose) (AtGrasp ?arm ?stirrer ?grasp) (AtConf ?arm ?conf1)
                           (Contains ?bowl @coffee) (Contains ?bowl @sugar)
                           (not (UnsafeControl ?arm ?control)))
        :effect (and (AtConf ?arm ?conf2) (CanMove ?arm) (Mixed ?bowl)
                     (not (AtConf ?arm ?conf1)))
    )

    (:action press-cook
        :parameters (?arm ?button ?obj ?stove ?link ?conf1 ?conf2 ?control)
        :precondition (and (IsPress ?arm ?button ?conf1 ?conf2 ?control) (IsButton ?button ?stove)
                           (Stackable ?obj ?stove ?link) (Stove ?stove ?link)
                           (On ?obj ?stove ?link) (Empty ?arm) (AtConf ?arm ?conf1)
                           (not (UnsafeControl ?arm ?control)))
        :effect (and (AtConf ?arm ?conf2) (Cooked ?obj) (CanMove ?arm)
                     ; (forall (?obj) (when (On ?obj ?stove) (Cooked ?obj))) ; TODO: implement fluent conditional effects
                     (not (AtConf ?arm ?conf1)))
    )

    ; Derived predicates
    (:derived (Grasped ?arm ?obj)
        (exists (?grasp) (and (IsArm ?arm) (IsGrasp ?obj ?grasp)
                              (AtGrasp ?arm ?obj ?grasp)))
    )
    (:derived (Holding ?obj)
        (exists (?arm) (and (IsArm ?arm) (Graspable ?obj)
                            (Grasped ?arm ?obj)))
    )
    (:derived (HoldingType ?ty)
        (exists (?obj) (and (IsType ?obj ?ty)
                            (Holding ?obj)))
    )

    (:derived (Supporting ?surface)
        (exists (?obj ?link) (and (Stackable ?obj ?surface ?link)
                                  (On ?obj ?surface ?link)))
    )
    (:derived (On ?obj ?surface ?link)
        (exists (?pose ?pose2) (and (IsSupported ?obj ?pose ?surface ?pose2 ?link)
                                    (AtPose ?obj ?pose))) ; (AtPose ?surface ?pose2)
    )
    (:derived (InRegion ?obj ?region)
        (exists (?pose) (and (Contained ?obj ?pose ?region)
                             (AtPose ?obj ?pose)))
    )

    ; Other arm at config w & w/o a held object
    (:derived (UnsafeControl ?arm ?control)
        ; Both of these at the same time makes a ton of axioms (particularly when omitting IsControl)
        (or ; TODO: move IsControl outside of the or
            (exists (?arm2 ?conf2) (and (IsControl ?arm ?control) (IsConf ?arm2 ?conf2)
                                        (ControlConfCollision ?arm ?control ?arm2 ?conf2) (not (= ?arm ?arm2))
                                        (AtConf ?arm2 ?conf2)))
            ; TODO: currently not counting anything that is held
            (exists (?obj ?pose) (and (IsControl ?arm ?control) (IsPose ?obj ?pose)
                                      (ControlPoseCollision ?arm ?control ?obj ?pose) (Movable ?obj)
                                      (AtPose ?obj ?pose)))
        )
    )

    ;(:derived (UnsafePose ?obj ?pose)
    ;    (exists (?obj2 ?pose2) (and (IsPose ?obj ?pose) (IsPose ?obj2 ?pose2)
    ;                                (PosePoseCollision ?obj ?pose ?obj2 ?pose2) (Movable ?obj2) (not (= ?obj ?obj2))
    ;                                (AtPose ?obj2 ?pose2)))
    ;)

    (:derived (UnsafeConf ?arm ?conf)
        (exists (?arm2 ?conf2) (and (IsConf ?arm ?conf) (IsConf ?arm2 ?conf2)
                                    (ConfConfCollision ?arm ?conf ?arm2 ?conf2) (not (= ?arm ?arm2))
                                    (AtConf ?arm2 ?conf2)))
    )

    ; TODO: could just do world pose
    ; https://github.mit.edu/mtoussai/KOMO-stream/blob/master/03-Caelans-pddlstreamExample/domain.pddl
)